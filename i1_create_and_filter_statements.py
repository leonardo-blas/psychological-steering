import argparse
import sqlite3
import subprocess
import shlex
from pathlib import Path
from tqdm.auto import tqdm
from helpers import normalize_table_name, table_has_enough


CONFIG = {
    "concepts_db": "data/concepts.db",
    "samples_per_label": 35000,
    "keep_per_label": 500,
}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Submit create/filter statement jobs for all concepts in a mode-specific table.",
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--vector",
        action="store_true",
        help="Use vector_concepts and vector statement DBs.",
    )
    group.add_argument(
        "--classifier",
        action="store_true",
        help="Use classifier_concepts and classifier statement DBs.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    db_path = Path(CONFIG["concepts_db"])
    mode = "vector" if args.vector else "classifier"
    concepts_table = f"{mode}_concepts"
    raw_db = f"data/raw_{mode}_statements.db"
    out_db = f"data/{mode}_statements.db"
    mode_flag = f"--{mode}"

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT concept, phrase FROM {concepts_table}")
        rows = cur.fetchall()

    for concept, phrase in tqdm(
        rows,
        desc="Concepts (statements)",
        unit="concept",
        leave=True,
    ):
        table = normalize_table_name(concept)

        expected_final = 2 * CONFIG["keep_per_label"]
        try:
            if table_has_enough(out_db, table, expected_final):
                continue
        except FileNotFoundError:
            pass

        expected_raw = 2 * CONFIG["samples_per_label"]
        try:
            have_raw = table_has_enough(raw_db, table, expected_raw)
        except FileNotFoundError:
            have_raw = False

        c_quoted = shlex.quote(concept)
        p_quoted = shlex.quote(phrase)

        if have_raw:
            wrapped_cmd = (
                "set -euo pipefail; "
                f"python3 -u 2_filter_statements.py -c {c_quoted} -p {p_quoted} {mode_flag}"
            )
        else:
            wrapped_cmd = (
                "set -euo pipefail; "
                f"python3 -u 1_create_statements.py -c {c_quoted} -p {p_quoted} {mode_flag}; "
                f"python3 -u 2_filter_statements.py -c {c_quoted} -p {p_quoted} {mode_flag}"
            )

        subprocess.run(
            [
                "sbatch",
                "-t",
                "48:00:00",
                "--gpus=1",
                "--constraint=a40",
                "--mem=32G",
                "--partition=gpu",
                "--output=logs/slurm-%j.log",
                "--error=logs/slurm-%j.log",
                "--wrap",
                wrapped_cmd,
            ],
            check=True,
        )
        print(f"Submitted job for {concept}.")


if __name__ == "__main__":
    main()
