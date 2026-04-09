import argparse
import json
import sqlite3
from pathlib import Path
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from helpers import normalize_table_name, seed_all


CONFIG = {
    "src_db": "data/statements.db",
    "seed": 42,
    "expected_rows": 1000,
    "system_text": "You are a person.",
    "user_template": "Tell me about yourself.",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("-c", "--concept", required=True)
    ap.add_argument("-q", "--quantize", action="store_true")
    return ap.parse_args()


def get_output_dir(model_name: str, concept: str) -> Path:
    model_dir = model_name.split("/")[-1]
    table = normalize_table_name(concept)
    return Path("vectors") / model_dir / table


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


def build_texts(rows, tokenizer):
    texts = []
    prefix_char_lens = []
    answer_char_lens = []
    labels = []

    for statement, raw_label in rows:
        label = int(raw_label)
        user = CONFIG["user_template"]
        assistant = statement

        prefix = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": CONFIG["system_text"]},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        texts.append(prefix + assistant)
        prefix_char_lens.append(len(prefix))
        answer_char_lens.append(len(assistant))
        labels.append(label)

    return texts, prefix_char_lens, answer_char_lens, labels


@torch.no_grad()
def extract_batch_states(batch, tokenizer, model, prefix_char_lens, answer_char_lens, start_idx):
    device = model.device
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
    hs = out.hidden_states
    num_layers = len(hs) - 1
    batch_size = input_ids.size(0)
    hidden_size = hs[-1].size(-1)
    reps = torch.empty(batch_size, num_layers, hidden_size, device="cpu")

    for b in range(batch_size):
        global_idx = start_idx + b
        start_char = prefix_char_lens[global_idx]
        end_char = start_char + answer_char_lens[global_idx]

        attn_row = attention_mask[b]
        offs_row = offsets[b]
        token_indices = []

        for t in range(attn_row.size(0)):
            if attn_row[t].item() == 0:
                continue
            token_start = int(offs_row[t][0].item())
            if start_char <= token_start < end_char:
                token_indices.append(t)

        if not token_indices:
            idxs = attn_row.nonzero(as_tuple=False).squeeze(-1)
            token_indices.append(idxs[-1].item())

        index_tensor = torch.tensor(token_indices, device=device)

        for layer in range(num_layers):
            layer_states = hs[layer + 1][b]
            reps[b, layer] = layer_states.index_select(0, index_tensor).mean(dim=0).cpu()

    return reps


@torch.no_grad()
def collect_reps_and_class_sums(
    texts,
    labels,
    tokenizer,
    model,
    batch_size,
    prefix_char_lens,
    answer_char_lens,
):
    pos_sum = None
    neg_sum = None
    pos_count = 0
    neg_count = 0
    rep_batches = []

    for i in tqdm(
        range(0, len(texts), batch_size),
        desc="Extracting activations + class means",
        unit="batch",
        leave=False,
    ):
        batch = texts[i : i + batch_size]
        reps = extract_batch_states(
            batch,
            tokenizer,
            model,
            prefix_char_lens,
            answer_char_lens,
            i,
        )
        rep_batches.append(reps)
        batch_labels = torch.tensor(labels[i : i + len(batch)], dtype=torch.bool)

        if pos_sum is None:
            pos_sum = torch.zeros_like(reps[0])
            neg_sum = torch.zeros_like(reps[0])

        if batch_labels.any():
            pos_sum += reps[batch_labels].sum(dim=0)
            pos_count += int(batch_labels.sum().item())

        neg_mask = ~batch_labels
        if neg_mask.any():
            neg_sum += reps[neg_mask].sum(dim=0)
            neg_count += int(neg_mask.sum().item())

    if pos_count == 0 or neg_count == 0:
        raise ValueError("Need both classes (0 and 1).")

    return torch.cat(rep_batches, dim=0), pos_sum / pos_count, neg_sum / neg_count


def save_vectors(mu1: torch.Tensor, mu0: torch.Tensor, out_dir: Path):
    vectors = mu1 - mu0
    norms = torch.linalg.norm(vectors, dim=1)
    if torch.any(norms <= 0):
        bad_layer = int(torch.nonzero(norms <= 0, as_tuple=False)[0].item())
        raise ValueError(f"Zero MeanDiff norm at layer {bad_layer}.")

    units = vectors / norms.unsqueeze(1)
    midpoints = 0.5 * (mu0 + mu1)

    out_dir.mkdir(parents=True, exist_ok=True)
    for layer in range(vectors.size(0)):
        torch.save(vectors[layer], out_dir / f"layer_{layer}_raw.pt")
        torch.save(units[layer], out_dir / f"layer_{layer}.pt")

    return units, midpoints


def collect_distances_from_reps(reps, labels, units, midpoints):
    per_layer = {
        layer: {
            0: {"sum": 0.0, "count": 0, "closest": None, "furthest": None},
            1: {"sum": 0.0, "count": 0, "closest": None, "furthest": None},
        }
        for layer in range(units.size(0))
    }

    for rep, label in tqdm(
        zip(reps, labels),
        total=len(labels),
        desc="Distances from retained activations",
        unit="text",
        leave=False,
    ):
        centered = rep - midpoints
        alphas = torch.sum(centered * units, dim=1)

        for layer, alpha in enumerate(alphas.tolist()):
            stats = per_layer[layer][int(label)]
            stats["sum"] += float(alpha)
            stats["count"] += 1

            if stats["closest"] is None or abs(alpha) < abs(stats["closest"]):
                stats["closest"] = float(alpha)
            if stats["furthest"] is None or abs(alpha) > abs(stats["furthest"]):
                stats["furthest"] = float(alpha)

    distances = {}
    for layer, layer_stats in per_layer.items():
        distances[str(layer)] = {}
        for cls, stats in layer_stats.items():
            if stats["count"] == 0:
                continue
            distances[str(layer)][str(cls)] = {
                "closest": stats["closest"],
                "centroid": stats["sum"] / stats["count"],
                "furthest": stats["furthest"],
            }

    return distances


def main():
    args = parse_args()
    seed_all(CONFIG["seed"])

    table = normalize_table_name(args.concept)
    out_dir = get_output_dir(args.model, args.concept)

    if (out_dir / "distances.json").exists():
        return

    rows = fetch_rows(table)

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant_cfg if args.quantize else None,
        low_cpu_mem_usage=True,
    ).eval()

    texts, prefix_char_lens, answer_char_lens, labels = build_texts(rows, tokenizer)

    reps, mu1, mu0 = collect_reps_and_class_sums(
        texts,
        labels,
        tokenizer,
        model,
        args.batch_size,
        prefix_char_lens,
        answer_char_lens,
    )
    units, midpoints = save_vectors(mu1, mu0, out_dir)

    distances = collect_distances_from_reps(reps, labels, units, midpoints)

    with open(out_dir / "distances.json", "w") as f:
        json.dump(distances, f, indent=2)


if __name__ == "__main__":
    main()
