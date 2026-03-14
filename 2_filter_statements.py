import argparse
import re
import sqlite3
import sys
import torch
from tqdm.auto import tqdm
from helpers import (
    seed_all,
    embed_batch,
    init_embed_model,
    init_fluency_model,
    fluency_filter_batch,
    normalize_table_name,
    table_has_enough,
)

CONFIG = {
    "seed": 42,
    "keep_per_label": 500,
    "batch_embed": 4096,
    "similarity_threshold": 0.9,
    "allowed": re.compile(r"^[A-Za-z0-9\s\.',]+$"),
    "fluency_batch": 4096,
    "fluency_threshold": 0.95,
}


def only_allowed(s: str) -> bool:
    return bool(CONFIG["allowed"].match(s or ""))


def dedup_on_gpu(rows, embed_tok, embed_model, desc):
    dedup_rows = []
    kept_embs = None
    if not rows:
        return dedup_rows
    total = len(rows)
    batch_size = CONFIG["batch_embed"]
    n_batches = (total + batch_size - 1) // batch_size
    for start in tqdm(
        range(0, total, batch_size),
        desc=desc,
        total=n_batches,
    ):
        end = min(start + batch_size, total)
        batch = rows[start:end]
        texts = [row["statement"] for row in batch]
        emb_batch = embed_batch(embed_tok, embed_model, texts)
        for i in range(emb_batch.size(0)):
            emb_vec = emb_batch[i : i + 1]
            if kept_embs is None:
                kept_embs = emb_vec.clone()
                dedup_rows.append(batch[i])
                continue
            sims = torch.matmul(emb_vec, kept_embs.transpose(0, 1))
            max_sim = sims.max().item()
            if max_sim >= CONFIG["similarity_threshold"]:
                continue
            kept_embs = torch.cat([kept_embs, emb_vec.clone()], dim=0)
            dedup_rows.append(batch[i])
    return dedup_rows


def filter_with_fluency(flu_tok, flu_model, rows, needed, desc):
    selected = []
    if not rows or needed <= 0:
        return selected
    total = len(rows)
    batch_size = CONFIG["fluency_batch"]
    n_batches = (total + batch_size - 1) // batch_size
    for start in tqdm(
        range(0, total, batch_size),
        desc=desc,
        total=n_batches,
    ):
        end = min(start + batch_size, total)
        batch = rows[start:end]
        texts = [row["statement"] for row in batch]
        keep_mask = fluency_filter_batch(
            flu_tok,
            flu_model,
            texts,
            CONFIG["fluency_threshold"],
        )
        for row, keep in zip(batch, keep_mask):
            if not keep:
                continue
            selected.append(row)
            if len(selected) >= needed:
                return selected
    return selected


def parse_args():
    ap = argparse.ArgumentParser(
        description="Filter raw statements with embed-dedup + fluency.",
    )
    ap.add_argument(
        "-c",
        "--concept",
        required=True,
        help='Human concept name (can contain spaces), e.g. "emotional control". '
        "Will be mapped to an underscore table name to match 1_create_statements.py.",
    )
    ap.add_argument(
        "-p",
        "--phrase",
        required=True,
        help="Phrase that was used to generate the statements (unused here, kept for CLI compatibility).",
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--classifier",
        action="store_true",
        help="Read/write classifier statement DBs.",
    )
    group.add_argument(
        "--vector",
        action="store_true",
        help="Read/write vector statement DBs.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    seed_all(CONFIG["seed"])
    concept_human = args.concept
    table = normalize_table_name(concept_human)
    expected_total = 2 * CONFIG["keep_per_label"]
    raw_sqlite_path = (
        "data/raw_classifier_statements.db"
        if args.classifier
        else "data/raw_vector_statements.db"
    )
    out_sqlite_path = (
        "data/classifier_statements.db"
        if args.classifier
        else "data/vector_statements.db"
    )

    try:
        if table_has_enough(out_sqlite_path, table, expected_total):
            return
    except FileNotFoundError:
        pass

    with sqlite3.connect(raw_sqlite_path) as rc:
        cur = rc.cursor()
        cur.execute(f"SELECT statement, label FROM {table}")
        rows = []
        fetched = cur.fetchall()
        for s, lbl in fetched:
            statement = (s or "").strip()
            if not statement:
                continue
            if not only_allowed(statement):
                continue
            row = {
                "statement": statement,
                "label": int(lbl),
            }
            rows.append(row)

    if not rows:
        sys.exit(0)

    pos_rows = []
    neg_rows = []
    for r in rows:
        if r["label"] == 1:
            pos_rows.append(r)
        else:
            neg_rows.append(r)

    embed_tok, embed_model = init_embed_model()

    pos_dedup = dedup_on_gpu(
        pos_rows,
        embed_tok,
        embed_model,
        desc="embed+dedup (pos)",
    )
    neg_dedup = dedup_on_gpu(
        neg_rows,
        embed_tok,
        embed_model,
        desc="embed+dedup (neg)",
    )

    del embed_model
    del embed_tok
    torch.cuda.empty_cache()

    flu_tok, flu_model = init_fluency_model()

    pos_top = filter_with_fluency(
        flu_tok,
        flu_model,
        pos_dedup,
        CONFIG["keep_per_label"],
        desc="Fluency (pos)",
    )
    neg_top = filter_with_fluency(
        flu_tok,
        flu_model,
        neg_dedup,
        CONFIG["keep_per_label"],
        desc="Fluency (neg)",
    )

    del flu_model
    del flu_tok
    torch.cuda.empty_cache()

    with sqlite3.connect(out_sqlite_path) as oc:
        c = oc.cursor()
        c.execute(f"DROP TABLE IF EXISTS {table}")
        c.execute(
            f"""
            CREATE TABLE {table} (
                statement TEXT NOT NULL,
                label INTEGER NOT NULL
            )
        """
        )
        rows_to_write = []
        for r in pos_top:
            rows_to_write.append((r["statement"], r["label"]))
        for r in neg_top:
            rows_to_write.append((r["statement"], r["label"]))
        c.executemany(
            f"INSERT INTO {table} (statement,label) VALUES (?,?)",
            rows_to_write,
        )
        oc.commit()

        c.execute(f"SELECT COUNT(*) FROM {table}")
        actual_total = c.fetchone()[0]

    if actual_total != expected_total:
        raise ValueError(
            f"The filtering process for '{concept_human}' only yielded {actual_total}/{expected_total} statements."
        )


if __name__ == "__main__":
    main()
