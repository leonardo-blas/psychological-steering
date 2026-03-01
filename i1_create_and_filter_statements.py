import sqlite3
import subprocess
import shlex
from pathlib import Path
from tqdm.auto import tqdm
from helpers import normalize_table_name, table_has_enough


CONFIG = {
    "concepts_db": "data/concepts.db",
    "raw_db": "data/raw_statements.db",       # must match 1_create_statements.py
    "out_db": "data/statements.db",           # must match 2_filter_statements.py
    "samples_per_label": 35000,               # must match CONFIG in 1_create_statements.py
    "keep_per_label": 500,                    # must match CONFIG in 2_filter_statements.py
}


def main():
    db_path = Path(CONFIG["concepts_db"])

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT concept, phrase FROM concepts")
        rows = cur.fetchall()

    for concept, phrase in tqdm(
        rows,
        desc="Concepts (statements)",
        unit="concept",
        leave=True,
    ):
#        if concept in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
#            continue
        table = normalize_table_name(concept)

        expected_final = 2 * CONFIG["keep_per_label"]
        if table_has_enough(CONFIG["out_db"], table, expected_final):
            # final table already done, skip this concept
            continue

        expected_raw = 2 * CONFIG["samples_per_label"]
        have_raw = table_has_enough(CONFIG["raw_db"], table, expected_raw)

        c_quoted = shlex.quote(concept)
        p_quoted = shlex.quote(phrase)

        if have_raw:
            wrapped_cmd = (
                "set -euo pipefail; "
                f"python3 -u 2_filter_statements.py -c {c_quoted} -p {p_quoted}"
            )
        else:
            wrapped_cmd = (
                "set -euo pipefail; "
                f"python3 -u 1_create_statements.py -c {c_quoted} -p {p_quoted}; "
                f"python3 -u 2_filter_statements.py -c {c_quoted} -p {p_quoted}"
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

