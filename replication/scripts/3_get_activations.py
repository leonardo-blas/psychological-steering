import argparse
import json
import sqlite3
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm.auto import tqdm
from helpers import seed_all, normalize_table_name, table_has_enough


CONFIG = {
    "src_db": "../../data/vector_statements.db",
    "seed": 42,
    "expected_rows": 1000,
    "system_text": "You are a person.",
    "user_template_b": (
        "Answer with Yes or No: Does the following statement accurately describe you?\n"
        "Statement: {s}\n"
        "Answer:"
    ),
    "user_template_s": "Tell me about yourself.",
    "yes_token": "Yes",
    "no_token": "No",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("-c", "--concept", required=True)
    ap.add_argument(
        "--mode",
        required=True,
        choices=["b", "s"],  # b = Yes/No, s = statement
    )
    ap.add_argument("-q", "--quantize", action="store_true")
    return ap.parse_args()


def get_dest_db_path(model_name: str, mode: str) -> str:
    model_short = model_name.split("/")[-1]
    if mode == "b":
        subdir = "binary_choice"
    else:
        subdir = "statement"
    return f"../../data/{subdir}_activations/{model_short}.db"


def fetch_rows(table: str):
    with sqlite3.connect(CONFIG["src_db"]) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT statement, label FROM {table}")
        rows = cur.fetchall()
    if len(rows) != CONFIG["expected_rows"]:
        raise ValueError(
            f"{table} has {len(rows)} rows, expected {CONFIG['expected_rows']}."
        )
    return rows


def build_texts(rows, mode: str, tokenizer):
    pairs = []
    statements = []
    labels = []
    sys_text = CONFIG["system_text"]

    for s, lbl in rows:
        if mode == "b":
            user = CONFIG["user_template_b"].format(s=s)
            assistant = CONFIG["yes_token"] if int(lbl) == 1 else CONFIG["no_token"]
        else:  # mode == "s"
            user = CONFIG["user_template_s"]
            assistant = s  # statement already has final period
        pairs.append((sys_text, user, assistant))
        statements.append(s)
        labels.append(int(lbl))

    texts = []
    prefix_char_lens = []
    answer_char_lens = []

    for sys_text, usr, asst in pairs:
        chat = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": usr},
        ]
        prefix = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        text = prefix + asst
        texts.append(text)
        prefix_char_lens.append(len(prefix))
        answer_char_lens.append(len(asst))

    return texts, prefix_char_lens, answer_char_lens, statements, labels


@torch.no_grad()
def extract_states(
    texts,
    tokenizer,
    model,
    batch_size,
    prefix_char_lens,
    answer_char_lens,
    mode: str,
):
    device = model.device
    L = model.config.num_hidden_layers
    feats = []

    for i in tqdm(
        range(0, len(texts), batch_size),
        desc="Extracting activations",
        unit="batch",
        leave=False,
    ):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding="longest",
            truncation=False,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        offsets = enc["offset_mapping"]

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.hidden_states  # list length L+1
        B = input_ids.size(0)
        D = hs[-1].size(-1)
        chunk = torch.empty(B, L, D, device="cpu")

        for b in range(B):
            global_idx = i + b
            start_char = prefix_char_lens[global_idx]
            end_char = start_char + answer_char_lens[global_idx]

            attn_row = attention_mask[b]
            offs_row = offsets[b]

            token_indices = []
            for t in range(attn_row.size(0)):
                if attn_row[t].item() == 0:
                    continue
                start = int(offs_row[t][0].item())
                if start >= start_char and start < end_char:
                    token_indices.append(t)

            # If somehow nothing matches the answer span, fall back to last non-pad
            if not token_indices:
                idxs = attn_row.nonzero(as_tuple=False).squeeze(-1)
                token_indices.append(idxs[-1].item())

            # IMPORTANT:
            # We DO NOT drop token_indices[0] anymore.
            # So the very first answer token (e.g., "I") is always included.

            index_tensor = torch.tensor(token_indices, device=device)

            for l in range(L):
                layer_states = hs[l + 1][b]  # skip embeddings
                selected = layer_states.index_select(0, index_tensor)
                rep = selected.mean(dim=0)
                chunk[b, l] = rep.cpu()

        feats.append(chunk)

    return torch.cat(feats, dim=0)


def main():
    args = parse_args()
    seed_all(CONFIG["seed"])

    model_name = args.model
    batch_size = args.batch_size
    concept_human = args.concept
    mode = args.mode

    table = normalize_table_name(concept_human)
    dest_db_path = get_dest_db_path(model_name, mode)

    if table_has_enough(dest_db_path, table, CONFIG["expected_rows"]):
        return

    rows = fetch_rows(table)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
        quantization_config=quant_cfg if args.quantize else None,
        low_cpu_mem_usage=True,
    ).eval()

    texts, prefix_lens, answer_lens, statements, labels = build_texts(
        rows, mode, tokenizer
    )

    feats = extract_states(
        texts,
        tokenizer,
        model,
        batch_size,
        prefix_lens,
        answer_lens,
        mode,
    )

    Path(dest_db_path).parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(dest_db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                statement TEXT NOT NULL,
                label INTEGER NOT NULL,
                activations TEXT NOT NULL
            )
            """
        )

        data = []
        for s, lbl, feat in zip(statements, labels, feats):
            payload = json.dumps(feat.tolist())
            data.append((s, lbl, payload))

        cur.executemany(
            f"INSERT INTO {table} (statement, label, activations) VALUES (?, ?, ?)",
            data,
        )
        conn.commit()


if __name__ == "__main__":
    main()
