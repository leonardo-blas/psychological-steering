import json
import re
import sqlite3
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging as hf_logging
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from helpers import (
    seed_all,
    format_atomic10x_head,
    ABS_SYSTEM_PROMPT,
    ABSOLUTE_PROMPT_WO_REF,
    rubric,
)

CONFIG = {
    "db_path": "data/heads.db",
    "atomic_path": "data/ATOMIC10X.jsonl",
    "seed": 42,
    # ATOMIC10X filter.
    "min_p_valid": 0.99,
    # Deduplication filter.
    "embed_model_id": "Qwen/Qwen3-Embedding-0.6B",
    "embed_batch": 4096,
    "dedup_cos_threshold": 0.9,
    # Readability and coherence filter.
    # Same decoding params as in the Prometheus 2 paper.
    "prometheus_model_id": "prometheus-eval/prometheus-7b-v2.0",
    "prometheus_batch": 128,
    "prometheus_temperature": 1.0,
    "prometheus_top_p": 0.9,
    "prometheus_repetition_penalty": 1.03,
    # Same generation params as in the Prometheus 2 paper.
    "prometheus_max_tokens": 1024,
    "prometheus_instruction": "Write a short and realistic sentence.",
}
hf_logging.set_verbosity_error()


def load_atomic_rows():
    rows = []
    with open(CONFIG["atomic_path"], "r") as f:
        for line in f:
            ex = json.loads(line)
            p_valid = ex.get("p_valid_model", 0.0)
            if p_valid < CONFIG["min_p_valid"]:
                continue
            head = ex["head"]
            row = {"head": head}
            rows.append(row)
    return rows


def init_embed_model():
    tok = AutoTokenizer.from_pretrained(CONFIG["embed_model_id"], padding_side="left")
    model = AutoModel.from_pretrained(
        CONFIG["embed_model_id"],
        dtype=torch.bfloat16,
    )
    model.to("cuda")
    model.eval()
    return tok, model

@torch.no_grad()
def embed_batch(embed_tok, embed_model, texts):
    x = embed_tok(
        texts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    x = x.to(embed_model.device)
    x = embed_model(**x)
    x = x.last_hidden_state[:, -1]
    return F.normalize(x, p=2, dim=1)


def init_prometheus():
    llm = LLM(
        CONFIG["prometheus_model_id"],
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        temperature=CONFIG["prometheus_temperature"],
        top_p=CONFIG["prometheus_top_p"],
        max_tokens=CONFIG["prometheus_max_tokens"],
        repetition_penalty=CONFIG["prometheus_repetition_penalty"],
    )
    return llm, sampling_params


def prometheus_filter_batch(llm, sampling_params, texts):
    prompts = []
    for text in texts:
        prompt = ABS_SYSTEM_PROMPT
        prompt = prompt + "\n\n"
        prompt = prompt + ABSOLUTE_PROMPT_WO_REF.format(
            instruction=CONFIG["prometheus_instruction"],
            response=text,
            rubric=rubric,
        )
        prompts.append(prompt)
    outputs = llm.generate(prompts, sampling_params)
    keep_mask = []
    for out in outputs:
        text = out.outputs[0].text
        tail = text.split("[RESULT]")[-1]
        digits = re.findall(r"\d", tail)
        if len(digits) == 1 and digits[0] in "45":
            keep_mask.append(True)
        else:
            keep_mask.append(False)
    return keep_mask


def main():
    seed_all(CONFIG["seed"])

    # pool all rows
    rows = load_atomic_rows()

    # lexical dedup on formatted heads
    lex_rows = []
    seen = set()
    for row in tqdm(rows, desc="lexical dedup"):
        text = format_atomic10x_head(row["head"])
        if text in seen:
            continue
        seen.add(text)
        lex_rows.append({"text": text})

    # embeddings + GPU dedup on lex-deduped strings
    embed_tok, embed_model = init_embed_model()
    dedup_rows = []
    kept_embs = None
    if lex_rows:
        total = len(lex_rows)
        batch_size = CONFIG["embed_batch"]
        n_batches = (total + batch_size - 1) // batch_size
        for start in tqdm(
            range(0, total, batch_size),
            desc="embed+dedup",
            total=n_batches,
        ):
            end = start + batch_size
            if end > total:
                end = total
            batch = lex_rows[start:end]
            texts = []
            for row in batch:
                texts.append(row["text"])
            emb_batch_tensor = embed_batch(
                embed_tok,
                embed_model,
                texts,
            )
            for i in range(emb_batch_tensor.size(0)):
                emb_vec = emb_batch_tensor[i : i + 1]
                if kept_embs is None:
                    kept_embs = emb_vec.clone()
                    emb_cpu = (
                        emb_vec.squeeze(0)
                        .detach()
                        .to(torch.float32)
                        .cpu()
                        .numpy()
                        .astype("float32")
                    )
                    row = batch[i]
                    dedup_rows.append(
                        {
                            "text": row["text"],
                            "embedding": emb_cpu,
                        }
                    )
                    continue
                sims = torch.matmul(
                    emb_vec,
                    kept_embs.transpose(0, 1),
                )
                max_sim = sims.max().item()
                if max_sim >= CONFIG["dedup_cos_threshold"]:
                    continue
                kept_embs = torch.cat(
                    [kept_embs, emb_vec.clone()],
                    dim=0,
                )
                emb_cpu = (
                    emb_vec.squeeze(0)
                    .detach()
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                    .astype("float32")
                )
                row = batch[i]
                dedup_rows.append(
                    {
                        "text": row["text"],
                        "embedding": emb_cpu,
                    }
                )
    del embed_model
    del embed_tok
    torch.cuda.empty_cache()

    # Prometheus-2 on pooled, deduped strings, then insert into single heads table
    llm, sampling_params = init_prometheus()
    with sqlite3.connect(CONFIG["db_path"]) as conn:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS heads")
        cur.execute(
            """
            CREATE TABLE heads (
                head TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
            """
        )
        rows = dedup_rows
        if rows:
            to_insert = []
            total = len(rows)
            batch_size = CONFIG["prometheus_batch"]
            n_batches = (total + batch_size - 1) // batch_size
            for start in tqdm(
                range(0, total, batch_size),
                desc="heads (Prometheus2)",
                total=n_batches,
            ):
                end = start + batch_size
                if end > total:
                    end = total
                batch = rows[start:end]
                texts = []
                for row in batch:
                    texts.append(row["text"])
                keep_mask = prometheus_filter_batch(
                    llm,
                    sampling_params,
                    texts,
                )
                for row, keep in zip(batch, keep_mask):
                    if not keep:
                        continue
                    text = row["text"]
                    emb = row["embedding"]
                    emb_bytes = emb.tobytes()
                    to_insert.append((text, emb_bytes))
            if to_insert:
                cur.executemany(
                    """
                    INSERT INTO heads (head, embedding)
                    VALUES (?, ?)
                    """,
                    to_insert,
                )
                conn.commit()


if __name__ == "__main__":
    main()

