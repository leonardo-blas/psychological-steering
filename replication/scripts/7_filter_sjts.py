import sqlite3
import torch
from tqdm.auto import tqdm
from helpers import seed_all, embed_batch, init_embed_model, init_fluency_model, fluency_filter_batch


CONFIG = {
    "raw_sqlite_path": "../data/raw_sjts.db",
    "out_sqlite_path": "../data/sjts.db",
    "seed": 42,
    "similarity_threshold": 0.875,
    "fluency_batch": 4096,
    "fluency_threshold": 0.95,
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


@torch.no_grad()
def build_conflict_adj(sjt_embs):
    n = sjt_embs.size(0)
    if n <= 1:
        return [0] * n
    sims = torch.matmul(sjt_embs, sjt_embs.transpose(0, 1)).float().cpu()
    all_mask = (1 << n) - 1
    adj = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i == j:
                continue
            if sims[i, j].item() > CONFIG["similarity_threshold"]:
                mask |= 1 << j
        adj[i] = mask & all_mask
    return adj


def first_set_bit_index(x):
    return (x & -x).bit_length() - 1


def greedy_independent_set_indices(adj, n, anchor_sims=None):
    active = (1 << n) - 1

    while True:
        edge_u = -1
        edge_v = -1
        for u in range(n):
            if ((active >> u) & 1) == 0:
                continue
            nbrs = adj[u] & active & ~(1 << u)
            if nbrs == 0:
                continue
            edge_u = u
            edge_v = first_set_bit_index(nbrs)
            break

        if edge_u < 0:
            break

        deg_u = (adj[edge_u] & active).bit_count()
        deg_v = (adj[edge_v] & active).bit_count()

        if deg_u > deg_v:
            drop = edge_u
        elif deg_v > deg_u:
            drop = edge_v
        else:
            sim_u = 0.0 if anchor_sims is None else anchor_sims[edge_u]
            sim_v = 0.0 if anchor_sims is None else anchor_sims[edge_v]
            if sim_u < sim_v:
                drop = edge_u
            elif sim_v < sim_u:
                drop = edge_v
            else:
                drop = max(edge_u, edge_v)

        active &= ~(1 << drop)

    indices = []
    for i in range(n):
        if (active >> i) & 1:
            indices.append(i)
    return indices


@torch.no_grad()
def compute_item_k_from_embs(sjt_embs):
    n = sjt_embs.size(0)
    if n <= 1:
        return n
    adj = build_conflict_adj(sjt_embs)
    return len(greedy_independent_set_indices(adj, n))


@torch.no_grad()
def select_topk_from_mis(rows, sjt_embs, item_emb, k):
    if not rows or k <= 0:
        return []

    anchor_sims = torch.matmul(
        sjt_embs, item_emb.transpose(0, 1)
    ).squeeze(1).float().cpu().tolist()
    adj = build_conflict_adj(sjt_embs)
    kept_indices = greedy_independent_set_indices(
        adj,
        len(rows),
        anchor_sims=anchor_sims,
    )
    order = sorted(kept_indices, key=lambda i: (-anchor_sims[i], i))

    selected_rows = []
    for i in order[:k]:
        selected_rows.append(rows[i])
    return selected_rows


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
        end = start + batch_size
        if end > total:
            end = total
        batch = rows[start:end]
        texts = [row["sjt"] for row in batch]
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


def main():
    seed_all(CONFIG["seed"])

    tables = get_tables(CONFIG["raw_sqlite_path"])

    tables_to_process = []
    with sqlite3.connect(CONFIG["raw_sqlite_path"]) as rc, sqlite3.connect(
        CONFIG["out_sqlite_path"]
    ) as oc:
        rc_cur = rc.cursor()
        oc_cur = oc.cursor()
        for table in tables:
            oc_cur.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name=?",
                (table,),
            )
            exists = oc_cur.fetchone() is not None
            if exists:
                oc_cur.execute(f"DROP TABLE IF EXISTS {table}")

            tables_to_process.append(table)
        oc.commit()

    if not tables_to_process:
        return

    dedup_by_table = {}
    flu_tok, flu_model = init_fluency_model()
    embed_tok, embed_model = init_embed_model()

    with sqlite3.connect(CONFIG["raw_sqlite_path"]) as rc:
        rc_cur = rc.cursor()
        for table in tables_to_process:
            rc_cur.execute(
                f"SELECT dimension, item, key, sjt FROM {table}"
            )
            rows = rc_cur.fetchall()

            items_dict = {}
            for dimension, item, key, sjt in rows:
                sjt = (sjt or "").strip()
                if not sjt:
                    continue
                item_key = (dimension, item, key)
                row = {
                    "dimension": dimension,
                    "item": item,
                    "key": key,
                    "sjt": sjt,
                }
                if item_key not in items_dict:
                    items_dict[item_key] = []
                items_dict[item_key].append(row)

            fluent_items_dict = {}
            item_keys = list(items_dict.keys())
            for idx, item_key in enumerate(item_keys):
                item_rows = items_dict[item_key]
                if not item_rows:
                    fluent_items_dict[item_key] = []
                    continue
                desc_flu = f"Fluency-pre {table} ({idx+1}/{len(item_keys)})"
                fluent_rows = filter_with_fluency(
                    flu_tok,
                    flu_model,
                    item_rows,
                    len(item_rows),
                    desc=desc_flu,
                )
                fluent_items_dict[item_key] = fluent_rows

            dedup_by_table[table] = {"K": 0, "items": {}}
            item_keys = list(fluent_items_dict.keys())
            k_values = []
            for idx, item_key in enumerate(item_keys):
                item_rows = fluent_items_dict[item_key]
                if not item_rows:
                    continue

                sjt_texts = []
                for row in item_rows:
                    sjt_texts.append(row["sjt"])
                sjt_embs = embed_batch(
                    embed_tok,
                    embed_model,
                    sjt_texts,
                )
                sjt_embs = sjt_embs.to("cuda")
                k_i = compute_item_k_from_embs(sjt_embs)
                k_values.append(k_i)
                desc_embed = f"k* {table} ({idx+1}/{len(item_keys)})"
                tqdm.write(f"{desc_embed}: k_i={k_i}, n_sjts={len(item_rows)}")

            if k_values:
                dedup_by_table[table]["K"] = min(k_values)
            table_k = dedup_by_table[table]["K"]
            tqdm.write(
                "table "
                f"{table}: K={table_k} (k_min at t={CONFIG['similarity_threshold']}) "
                f"from {len(k_values)} items"
            )

            for idx, item_key in enumerate(item_keys):
                item_rows = fluent_items_dict[item_key]
                if not item_rows:
                    dedup_by_table[table]["items"][item_key] = []
                    continue

                sjt_texts = []
                for row in item_rows:
                    sjt_texts.append(row["sjt"])
                sjt_embs = embed_batch(
                    embed_tok,
                    embed_model,
                    sjt_texts,
                )
                sjt_embs = sjt_embs.to("cuda")

                item_text = item_key[1]
                item_emb = embed_batch(
                    embed_tok,
                    embed_model,
                    [item_text],
                )
                item_emb = item_emb.to("cuda")

                selected_rows = select_topk_from_mis(
                    item_rows,
                    sjt_embs,
                    item_emb,
                    table_k,
                )
                dedup_by_table[table]["items"][item_key] = selected_rows

                desc_sel = f"select {table} ({idx+1}/{len(item_keys)})"
                tqdm.write(
                    f"{desc_sel}: selected={len(selected_rows)}/{table_k}, n_sjts={len(item_rows)}"
                )

    del embed_model
    del embed_tok
    del flu_model
    del flu_tok
    torch.cuda.empty_cache()

    with sqlite3.connect(CONFIG["out_sqlite_path"]) as oc:
        oc_cur = oc.cursor()
        for table in tables_to_process:
            oc_cur.execute(
                f"CREATE TABLE {table} ("
                "dimension TEXT,"
                "item TEXT,"
                "key INTEGER,"
                "sjt TEXT"
                ")"
            )

            table_info = dedup_by_table.get(table, {"K": 0, "items": {}})
            table_k = table_info["K"]
            item_dict = table_info["items"]
            item_keys = list(item_dict.keys())
            for idx, item_key in enumerate(item_keys):
                selected_rows = item_dict[item_key]
                if not selected_rows:
                    continue
                to_write = []
                for r in selected_rows:
                    to_write.append(
                        (
                            r["dimension"],
                            r["item"],
                            r["key"],
                            r["sjt"],
                        )
                    )
                oc_cur.executemany(
                    f"INSERT INTO {table} "
                    "(dimension, item, key, sjt) "
                    "VALUES (?, ?, ?, ?)",
                    to_write,
                )

            oc.commit()


if __name__ == "__main__":
    main()
