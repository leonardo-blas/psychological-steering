# ATOMIC10X.jsonl may be procured from https://github.com/peterwestai2/symbolic-knowledge-distillation/tree/main.

import json
import re
import sqlite3
import torch
from transformers.utils import logging as hf_logging
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from helpers import seed_all, embed_batch, init_embed_model


CONFIG = {
    "db_path": "../../data/heads.db",
    "atomic_path": "../../data/ATOMIC10X.jsonl",
    "seed": 42,
    # ATOMIC10X filter.
    "min_p_valid": 0.99,
    # Deduplication filter.
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
# From Prometeus-Eval.
ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
ABSOLUTE_PROMPT_WO_REF = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubrics:
{rubric}

###Feedback: """
# From FLASK.
rubric = (
    "[Is the response structured to promote readability and coherence? Does the response exhibit excellent organization?]\n"
    "Score 1: The response is completely unclear, making comprehension difficult.\n"
    "Score 2: The response has significant areas of ambiguity or disorganization, critically affecting reader comprehension.\n"
    "Score 3: The response contains some unclear components, or its organization could be improved.\n"
    "Score 4: The response is generally understandable but could be further optimized for readability.\n"
    "Score 5: The response is clear and well-organized, enabling the reader to effortlessly follow the content.\n"
)
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


def format_atomic10x_head(text: str) -> str:
    s = text
    name_x = "Alex" if "Alex" not in s else "Avery"
    name_y = "Brook" if "Brook" not in s else "Blake"
    name_z = "Charlie" if "Charlie" not in s else "Cameron"
    s = re.sub(r"\b[Pp]ersonX\b", name_x, s)
    s = re.sub(r"\b[Pp]ersonY\b", name_y, s)
    s = re.sub(r"\b[Pp]ersonZ\b", name_z, s)
    return s if s.endswith(".") else s + "."


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
