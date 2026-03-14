import re
import sqlite3
import numpy as np
import torch
from helpers import seed_all, init_embed_model
from openai import OpenAI
from tqdm.auto import tqdm


CONFIG = {
    "seed": 42,
    "heads_db": "../data/heads.db",
    "psych_db": "../data/inventories.db",
    "raw_sjts_db": "../data/raw_sjts.db",
    "sjts_per_item": 25,
    "embed_batch": 4096,
    "llm": "gpt-5.1",
    # Same decoding params as in Jiang et al.
    "top_p": 0.8,
    "temperature": 0.8,
    "max_new_tokens": 128,
    "llm_batch": 128,
    "system_prompt": (
        "We are creating interview questions for psychological studies.\n"
        "Given a sample situation and a behavioral tendency, create a "
        "scenario-based, story-like question to prompt an answer that "
        "would reveal the presence or lack of this tendency in a person. "
        "The output must be sentences in a single paragraph. "
        "The first sentence must be a very short, concrete, realistic, "
        "actionable, and setting-focused scenario description; it must "
        "be conceptually inspired by the sample situation but "
        "reformulated into a generic form that is natural and does not "
        "explicitly reveal the situation. "
        "The second sentence must be a very short, concrete, natural, "
        "and personal question about the scenario, e.g. 'What would you "
        "do?', 'How would you solve this?', 'What do you think about "
        "this?', etc. "
        "Both sentences must be framed around the person, not around a "
        "third party. "
        "Neither sentence may imply, assert, or hypothesize anything "
        "about the subject's character, mental state, physique, or "
        "physical state. "
        "Do not include any options or explanations."
    ),
    "user_prompt": (
        "Behavioral tendency: You {item}\n"
        "Situation: {head}\n"
        "Question: "
    ),
}


def get_tables(db_path):
    tables = []
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        rows = cur.fetchall()
    for row in rows:
        tables.append(row[0])
    return tables


def load_heads():
    with sqlite3.connect(CONFIG["heads_db"]) as conn:
        cur = conn.cursor()
        cur.execute("SELECT head, embedding FROM heads")
        rows = cur.fetchall()
    heads = []
    vectors = []
    for head, blob in rows:
        heads.append(head)
        vec = np.frombuffer(blob, dtype=np.float32)
        vectors.append(vec)
    matrix = np.vstack(vectors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    matrix = matrix / norms
    return heads, matrix


def encode_items_batch(texts, tokenizer, model, batch_size):
    embeddings = []
    idx = 0
    with torch.no_grad():
        while idx < len(texts):
            batch_texts = texts[idx : idx + batch_size]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                )
                for key in inputs:
                    inputs[key] = inputs[key].to("cuda")
                outputs = model(**inputs)
                hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                masked = hidden * mask
                summed = masked.sum(dim=1)
                counts = mask.sum(dim=1)
                counts[counts == 0] = 1
                pooled = summed / counts
            batch_emb = pooled.to(torch.float32).cpu().numpy()
            for j in range(batch_emb.shape[0]):
                emb = batch_emb[j]
                norm = np.linalg.norm(emb)
                if norm == 0.0:
                    embeddings.append(None)
                else:
                    emb = emb / norm
                    embeddings.append(emb)
            idx += batch_size
    return embeddings


def top_k_indices(sims, k):
    if sims.shape[0] <= k:
        return np.argsort(-sims)
    partial = np.argpartition(-sims, k - 1)[:k]
    ordered = partial[np.argsort(-sims[partial])]
    return ordered


def format_item_for_llm(item):
    item = item[0].lower() + item[1:]
    if not item.endswith("."):
        item += "."
    return item


def build_question_messages(item, head):
    item_llm = format_item_for_llm(item)
    system = CONFIG["system_prompt"]
    user = CONFIG["user_prompt"].format(item=item_llm, head=head)
    messages = []
    messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


def clean_sjt(text):
    s = text.strip()
    if len(s) >= 2 and (
        (s[0] == '"' and s[-1] == '"')
        or (s[0] == "“" and s[-1] == "”")
    ):
        s = s[1:-1].strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def ensure_raw_table(conn, table_name):
    cur = conn.cursor()
    cur.execute(
        f"CREATE TABLE IF NOT EXISTS {table_name} ("
        "dimension TEXT,"
        "item TEXT,"
        "key INTEGER,"
        "sjt TEXT"
        ")"
    )
    conn.commit()


def main():
    seed_all(CONFIG["seed"])
    all_psych_tables = get_tables(CONFIG["psych_db"])
    psych_tables = []
    with sqlite3.connect(CONFIG["psych_db"]) as psych_conn:
        psych_cur = psych_conn.cursor()
        for psych_table in all_psych_tables:
            psych_cur.execute(f"SELECT COUNT(*) FROM {psych_table}")
            n_items = psych_cur.fetchone()[0]
            if n_items == 0:
                continue
            psych_tables.append(psych_table)
    if not psych_tables:
        return
    embed_tokenizer, embed_model = init_embed_model()
    heads, heads_matrix = load_heads()
    psych_items = []
    all_texts = []
    with sqlite3.connect(CONFIG["psych_db"]) as psych_conn, sqlite3.connect(CONFIG["raw_sjts_db"]) as raw_conn:
        psych_cur = psych_conn.cursor()
        raw_cur = raw_conn.cursor()
        for psych_table in psych_tables:
            print(f"Processing table: {psych_table}", flush=True)
            psych_cur.execute(
                f"SELECT dimension, item, key FROM {psych_table}"
            )
            rows = psych_cur.fetchall()
            raw_cur.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name=?",
                (psych_table,),
            )
            exists = raw_cur.fetchone() is not None
            existing_counts = {}
            if exists:
                raw_cur.execute(
                    f"SELECT dimension, item, key, COUNT(*) "
                    f"FROM {psych_table} "
                    "GROUP BY dimension, item, key"
                )
                for d, it, k, n in raw_cur.fetchall():
                    existing_counts[(d, it, int(k))] = int(n)
            for dimension, item, key in rows:
                item_key = (dimension, item, int(key))
                existing_n = existing_counts.get(item_key, 0)
                needed = CONFIG["sjts_per_item"] - existing_n
                if needed <= 0:
                    continue
                rec = {
                    "table": psych_table,
                    "dimension": dimension,
                    "item": item,
                    "key": int(key),
                    "needed": int(needed),
                }
                psych_items.append(rec)
                all_texts.append(item)
    print(f"Total items: {len(all_texts)}", flush=True)
    item_embs = encode_items_batch(
        all_texts,
        embed_tokenizer,
        embed_model,
        CONFIG["embed_batch"],
    )
    del embed_model
    del embed_tokenizer
    torch.cuda.empty_cache()
    jobs = []
    needed_by_item = {}
    for idx, rec in enumerate(tqdm(psych_items, desc="Building SJT jobs")):
        item_vec = item_embs[idx]
        if item_vec is None:
            continue
        item_key = (
            rec["table"],
            rec["dimension"],
            rec["item"],
            int(rec["key"]),
        )
        needed_by_item[item_key] = int(rec["needed"])
        sims = heads_matrix @ item_vec
        k = min(int(rec["needed"]), sims.shape[0])
        indices = top_k_indices(sims, k)
        for h_idx in indices:
            job = {
                "table": rec["table"],
                "dimension": rec["dimension"],
                "item": rec["item"],
                "key": rec["key"],
                "head": heads[h_idx],
            }
            jobs.append(job)
    print(f"Total jobs: {len(jobs)}", flush=True)
    with sqlite3.connect(CONFIG["raw_sjts_db"]) as conn:
        for psych_table in psych_tables:
            ensure_raw_table(conn, psych_table)
    client = OpenAI()
    batch_size = CONFIG["llm_batch"]
    with sqlite3.connect(CONFIG["raw_sjts_db"]) as conn:
        cur = conn.cursor()
        current_item_key = None
        current_item_target = 0
        pending_rows = []

        def flush_current_item():
            nonlocal pending_rows
            if not pending_rows:
                return
            table_name = pending_rows[0][0]
            values = []
            for _, d, it, k, sjt in pending_rows:
                values.append((d, it, k, sjt))
            cur.executemany(
                f"INSERT INTO {table_name} (dimension, item, key, sjt) "
                "VALUES (?, ?, ?, ?)",
                values,
            )
            conn.commit()
            pending_rows = []

        for i in tqdm(
            range(0, len(jobs), batch_size),
            desc="Generating SJTs",
            unit="batch",
        ):
            batch = jobs[i : i + batch_size]
            for job in batch:
                messages = build_question_messages(
                    job["item"],
                    job["head"],
                )
                try:
                    completion = client.chat.completions.create(
                        model=CONFIG["llm"],
                        messages=messages,
                        temperature=CONFIG["temperature"],
                        top_p=CONFIG["top_p"],
                        max_completion_tokens=CONFIG["max_new_tokens"],
                        seed=CONFIG["seed"],
                    )
                except Exception as e:
                    print(f"OpenAI error: {repr(e)}", flush=True)
                    continue
                text = completion.choices[0].message.content
                if text is None:
                    continue
                sjt = clean_sjt(text)

                print("\n==============================", flush=True)
                print(f"ITEM: {job['item']}", flush=True)
                print(f"SITUATION: {job['head']}", flush=True)
                print("SJT:", flush=True)
                print(sjt, flush=True)
                print("==============================\n", flush=True)

                item_key = (
                    job["table"],
                    job["dimension"],
                    job["item"],
                    int(job["key"]),
                )
                if current_item_key is None:
                    current_item_key = item_key
                    current_item_target = needed_by_item.get(item_key, CONFIG["sjts_per_item"])
                elif item_key != current_item_key:
                    flush_current_item()
                    current_item_key = item_key
                    current_item_target = needed_by_item.get(item_key, CONFIG["sjts_per_item"])

                pending_rows.append(
                    (
                        job["table"],
                        job["dimension"],
                        job["item"],
                        job["key"],
                        sjt,
                    )
                )
                if len(pending_rows) >= current_item_target:
                    flush_current_item()

        flush_current_item()


if __name__ == "__main__":
    main()
