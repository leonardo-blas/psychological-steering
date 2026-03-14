import json
import sqlite3
from pathlib import Path

import joblib
from openai import OpenAI
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from helpers import (
    embed_texts,
    init_embed_model,
    normalize_table_name,
)


CONFIG = {
    "input_db": "../data/validation_inventories.db",
    "output_db": "../data/validation_inventories_labeled.db",
    "ours": {
        "output_suffix": "ours",
        "classifiers_dir": "classifiers",
        "batch_size": 1024,
    },
    "gpt": {
        "output_suffix": "gpt",
        "model": "gpt-5.1",
        "personality_system_prompt": "You are asked to evaluate the personality of a person based on their response.",
        "personality_user_prompt_template": (
            "First consider the situation provided below as the context."
            " Then consider the response and rate the degree of {trait} trait in the response "
            "from 1 to 5, where 1 means very low and 5 means very high.\n"
            "\nReturn **ONE LINE of JSON ONLY** in the form:\n"
            '\n{{"score":X}}\n\n'
            "Context: {situation}\n"
            "---\n"
            "Response: {response}\n"
            "---\n"
            "For your reference, a sample response with high score is: {high1}\n"
            "For your reference, a sample response with low score is: {low1}\n"
            "---\n"
            "JSON:"
        ),
    },
}


def process_item_text(item: str) -> str:
    item = (item or "").strip()
    if not item:
        return ""
    return item


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def list_source_tables(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        ORDER BY name
        """
    )
    return [row[0] for row in cur.fetchall()]


def load_rows_for_table(conn: sqlite3.Connection, source_table: str):
    rows = []
    qtable = quote_ident(source_table)
    cur = conn.cursor()
    cur.execute(f"SELECT key, concept, item FROM {qtable}")
    for idx, row in enumerate(cur.fetchall(), start=1):
        key_raw, concept, item_raw = row
        concept = (concept or "").strip()
        item_raw = (item_raw or "").strip()
        key_str = str(key_raw).strip()
        if not concept or not item_raw or key_str not in {"-1", "1"}:
            continue
        item = process_item_text(item_raw)
        if not item:
            continue
        rows.append(
            {
                "row_id": idx,
                "source_table": source_table,
                "concept": concept,
                "item": item,
                "label": 1 if key_str == "1" else 0,
            }
        )
    return rows


def existing_items(conn: sqlite3.Connection, table_name: str):
    cur = conn.cursor()
    cur.execute(f"SELECT concept, item FROM {quote_ident(table_name)}")
    return set(cur.fetchall())


def ensure_table_with_schema(
    conn: sqlite3.Connection,
    table_name: str,
    expected_cols,
    create_sql: str,
):
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    exists = cur.fetchone() is not None
    if exists:
        cur.execute(f'PRAGMA table_info("{table_name}")')
        existing_cols = [r[1] for r in cur.fetchall()]
        if existing_cols != expected_cols:
            raise RuntimeError(
                f"Existing table '{table_name}' has legacy schema. "
                f"Please run: DROP TABLE {table_name}; and rerun."
            )
    cur.execute(create_sql)
    conn.commit()


def binary_metrics(y_true, y_pred):
    total = len(y_true)
    if total == 0:
        return 0.0, 0.0
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    acc = 100.0 * float(correct) / float(total)
    f1_macro = 100.0 * float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return acc, f1_macro


def table_metrics_ours(conn: sqlite3.Connection, table_name: str, keys):
    cur = conn.cursor()
    cur.execute(f"SELECT concept, item, label, score FROM {quote_ident(table_name)}")
    key_set = set(keys)
    matched = []
    for concept, item, label, score in cur.fetchall():
        if (concept, item) in key_set:
            matched.append((int(label), float(score)))
    total = len(matched)
    if total == 0:
        return 0, 0.0, 0.0
    y_true = [label for label, _ in matched]
    y_pred = [1 if score > 0.5 else 0 for _, score in matched]
    acc, f1_macro = binary_metrics(y_true, y_pred)
    return total, acc, f1_macro


def table_metrics_gpt(conn: sqlite3.Connection, table_name: str, keys):
    cur = conn.cursor()
    cur.execute(f"SELECT concept, item, label, score FROM {quote_ident(table_name)}")
    key_set = set(keys)
    matched = []
    for concept, item, label, score in cur.fetchall():
        if (concept, item) in key_set:
            matched.append((int(label), int(score)))
    total = len(matched)
    if total == 0:
        return 0, 0.0, 0.0
    y_true = [label for label, _ in matched]
    y_pred = [1 if score > 3 else 0 for _, score in matched]
    acc, f1_macro = binary_metrics(y_true, y_pred)
    return total, acc, f1_macro


class GptEvaluator:
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI()

    def evaluate_personality(
        self,
        situation: str,
        response: str,
        trait: str,
        high1: str,
        low1: str,
        system_prompt: str,
        user_prompt_template: str,
    ):
        trait = trait.lower()
        prompt = user_prompt_template.format(
            situation=situation,
            response=response,
            trait=trait,
            high1=high1,
            low1=low1,
        )
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_completion_tokens=64,
        )
        raw = (completion.choices[0].message.content or "").strip()
        parsed = json.loads(raw)
        return raw, parsed


def run_ours():
    input_path = Path(CONFIG["input_db"])
    output_path = Path(CONFIG["output_db"])
    if not input_path.is_file() or input_path.stat().st_size == 0:
        raise RuntimeError(f"Input DB not found or empty: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embed_tok = None
    embed_model = None
    classifiers_dir = Path(CONFIG["ours"]["classifiers_dir"])
    clf_cache = {}

    total_rows = 0
    weighted_acc_sum = 0.0
    weighted_f1_macro_sum = 0.0
    total_skipped_existing = 0

    with sqlite3.connect(input_path) as src_conn, sqlite3.connect(output_path) as out_conn:
        source_tables = list_source_tables(src_conn)
        if not source_tables:
            raise RuntimeError(f"No tables found in input DB: {input_path}")

        for source_table in source_tables:
            rows = load_rows_for_table(src_conn, source_table)
            if not rows:
                print(f"Skipping {source_table}: no valid rows.")
                continue

            out_table = f"{source_table}_{CONFIG['ours']['output_suffix']}"
            ensure_table_with_schema(
                out_conn,
                out_table,
                ["concept", "item", "label", "score"],
                f"""
                CREATE TABLE IF NOT EXISTS "{out_table}" (
                    concept TEXT NOT NULL,
                    item TEXT NOT NULL,
                    label INTEGER NOT NULL,
                    score REAL NOT NULL,
                    UNIQUE(concept, item)
                )
                """,
            )

            already_done = existing_items(out_conn, out_table)
            rows_to_process = [
                row for row in rows if (row["concept"], row["item"]) not in already_done
            ]
            skipped_existing = len(rows) - len(rows_to_process)
            total_skipped_existing += skipped_existing

            if not rows_to_process:
                print(
                    f"Skipping {source_table}: all {len(rows)} items already processed "
                    f"in {out_table}."
                )
            else:
                if embed_tok is None or embed_model is None:
                    embed_tok, embed_model = init_embed_model()

                texts = [r["item"] for r in rows_to_process]
                X = embed_texts(
                    embed_tok=embed_tok,
                    embed_model=embed_model,
                    texts=texts,
                    batch_size=int(CONFIG["ours"]["batch_size"]),
                )

                inserts = []
                desc = (
                    f"Classifier-evaluating {source_table} "
                    f"({len(rows_to_process)} new, {skipped_existing} existing)"
                )
                for i, row in enumerate(tqdm(rows_to_process, desc=desc)):
                    concept_norm = normalize_table_name(row["concept"])
                    if concept_norm not in clf_cache:
                        clf_path = classifiers_dir / f"{concept_norm}.pkl"
                        if not clf_path.is_file():
                            raise FileNotFoundError(
                                f"Missing classifier for {row['concept']}: {clf_path}"
                            )
                        clf_cache[concept_norm] = joblib.load(clf_path)

                    clf = clf_cache[concept_norm]
                    prob = float(clf.predict_proba(X[i : i + 1])[0][1])
                    inserts.append(
                        (
                            row["concept"],
                            row["item"],
                            int(row["label"]),
                            prob,
                        )
                    )

                cur = out_conn.cursor()
                cur.executemany(
                    f"""
                    INSERT INTO {quote_ident(out_table)} (
                        concept, item, label, score
                    )
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(concept, item) DO UPDATE SET
                        concept=excluded.concept,
                        item=excluded.item,
                        label=excluded.label,
                        score=excluded.score
                    """,
                    inserts,
                )
                out_conn.commit()

            keys = [(row["concept"], row["item"]) for row in rows]
            total, acc, f1_macro = table_metrics_ours(out_conn, out_table, keys)
            print(f"[{out_table}] construct accuracy: {acc:.2f}% ({total} items)")
            print(f"[{out_table}] f1-macro: {f1_macro:.2f}%")
            print(
                f"[{out_table}] processed new items: {len(rows_to_process)}, "
                f"skipped existing: {skipped_existing}"
            )

            total_rows += total
            weighted_acc_sum += acc * float(total)
            weighted_f1_macro_sum += f1_macro * float(total)

    final_acc = weighted_acc_sum / float(total_rows) if total_rows else 0.0
    final_f1_macro = weighted_f1_macro_sum / float(total_rows) if total_rows else 0.0
    print(f"Overall construct accuracy: {final_acc:.2f}% ({total_rows} items)")
    print(f"Overall f1-macro: {final_f1_macro:.2f}%")
    print(f"Overall skipped existing items: {total_skipped_existing}")


def run_gpt():
    input_path = Path(CONFIG["input_db"])
    output_path = Path(CONFIG["output_db"])
    if not input_path.is_file() or input_path.stat().st_size == 0:
        raise RuntimeError(f"Input DB not found or empty: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_version = CONFIG["gpt"]["model"]
    evaluator = GptEvaluator(model=model_version)

    total_rows = 0
    weighted_construct_acc_sum = 0.0
    weighted_f1_macro_sum = 0.0
    total_skipped_existing = 0

    with sqlite3.connect(input_path) as src_conn, sqlite3.connect(output_path) as out_conn:
        source_tables = list_source_tables(src_conn)
        if not source_tables:
            raise RuntimeError(f"No tables found in input DB: {input_path}")

        for source_table in source_tables:
            rows = load_rows_for_table(src_conn, source_table)
            if not rows:
                print(f"Skipping {source_table}: no valid rows.")
                continue

            out_table = f"{source_table}_{CONFIG['gpt']['output_suffix']}"
            ensure_table_with_schema(
                out_conn,
                out_table,
                ["concept", "item", "label", "score"],
                f"""
                CREATE TABLE IF NOT EXISTS "{out_table}" (
                    concept TEXT NOT NULL,
                    item TEXT NOT NULL,
                    label INTEGER NOT NULL,
                    score INTEGER NOT NULL,
                    UNIQUE(concept, item)
                )
                """,
            )

            already_done = existing_items(out_conn, out_table)
            rows_to_process = [
                row for row in rows if (row["concept"], row["item"]) not in already_done
            ]
            skipped_existing = len(rows) - len(rows_to_process)
            total_skipped_existing += skipped_existing

            if not rows_to_process:
                print(
                    f"Skipping {source_table}: all {len(rows)} items already processed "
                    f"in {out_table}."
                )
            else:
                cur = out_conn.cursor()
                desc = (
                    f"Evaluating {source_table} with {model_version} "
                    f"({len(rows_to_process)} new, {skipped_existing} existing)"
                )
                for row in tqdm(rows_to_process, desc=desc):
                    try:
                        _, personality = evaluator.evaluate_personality(
                            situation=f"Personality inventory statement for construct: {row['concept']}",
                            response=row["item"],
                            trait=row["concept"],
                            high1="",
                            low1="",
                            system_prompt=CONFIG["gpt"]["personality_system_prompt"],
                            user_prompt_template=CONFIG["gpt"]["personality_user_prompt_template"],
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"Evaluation failed for table={source_table}, row={row['row_id']}: {repr(exc)}"
                        ) from exc

                    score = int(personality["score"])
                    if not (1 <= score <= 5):
                        raise ValueError(
                            "Out-of-range scores for table="
                            f"{source_table}, item={row['row_id']}: {personality}"
                        )

                    cur.execute(
                        f"""
                        INSERT INTO {quote_ident(out_table)} (
                            concept, item, label, score
                        )
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(concept, item) DO UPDATE SET
                            concept=excluded.concept,
                            item=excluded.item,
                            label=excluded.label,
                            score=excluded.score
                        """,
                        (
                            row["concept"],
                            row["item"],
                            int(row["label"]),
                            score,
                        ),
                    )
                    out_conn.commit()

            keys = [(row["concept"], row["item"]) for row in rows]
            total, construct_acc_pct, f1_macro_pct = table_metrics_gpt(out_conn, out_table, keys)
            print(
                f"[{out_table}] {model_version} construct accuracy: "
                f"{construct_acc_pct:.2f}% ({total} items)"
            )
            print(
                f"[{out_table}] {model_version} f1-macro: "
                f"{f1_macro_pct:.2f}%"
            )
            print(
                f"[{out_table}] processed new items: {len(rows_to_process)}, "
                f"skipped existing: {skipped_existing}"
            )

            total_rows += total
            weighted_construct_acc_sum += construct_acc_pct * float(total)
            weighted_f1_macro_sum += f1_macro_pct * float(total)

    overall_construct_acc = weighted_construct_acc_sum / float(total_rows) if total_rows else 0.0
    overall_f1_macro = weighted_f1_macro_sum / float(total_rows) if total_rows else 0.0
    print(
        f"Overall {model_version} construct accuracy: "
        f"{overall_construct_acc:.2f}% ({total_rows} items)"
    )
    print(
        f"Overall {model_version} f1-macro: "
        f"{overall_f1_macro:.2f}%"
    )
    print(f"Overall skipped existing items: {total_skipped_existing}")


def main():
    run_ours()
    run_gpt()


if __name__ == "__main__":
    main()
