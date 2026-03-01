import argparse
import os
import sqlite3
import subprocess
from tqdm.auto import tqdm
from helpers import normalize_table_name


CONFIG = {
    "concepts_db": "data/concepts.db",
    "commands": [
        'python3 -u 4_create_vectors_lr.py --mode b --model "$model" --concept "$c" -r l2 -i',
        'python3 -u 4_create_vectors_lr.py --mode b --model "$model" --concept "$c" -r l2',
        'python3 -u 4_create_vectors_lr.py --mode b --model "$model" --concept "$c" -r l1 -i',
        'python3 -u 4_create_vectors_lr.py --mode b --model "$model" --concept "$c" -r l1',

        'python3 -u 4_create_vectors_lr.py --mode s --model "$model" --concept "$c" -r l2 -i',
        'python3 -u 4_create_vectors_lr.py --mode s --model "$model" --concept "$c" -r l2',
        'python3 -u 4_create_vectors_lr.py --mode s --model "$model" --concept "$c" -r l1 -i',
        'python3 -u 4_create_vectors_lr.py --mode s --model "$model" --concept "$c" -r l1',
    ],
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    return ap.parse_args()


def load_concepts(db_path: str):
    concepts = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
#    concepts = []
#    with sqlite3.connect(db_path) as conn:
#        cur = conn.cursor()
#        cur.execute("SELECT concept FROM concepts")
#        for (c,) in cur.fetchall():
#            c = str(c).strip()
#            if c:
#                concepts.append(c)
    return concepts


def submit(cmd: str):
    subprocess.run(
        [
            "sbatch",
            "-t",
            "48:00:00",
            "--mem=8G",
            "--partition=main,epyc-64",
            "--output=logs/slurm-%j.log",
            "--error=logs/slurm-%j.log",
            "--wrap",
            cmd,
        ],
        check=True,
    )


def main():
    args = parse_args()
    model_dir = args.model.split("/")[-1]
    concepts = load_concepts(CONFIG["concepts_db"])

    needed_files = [
        "metrics.pdf",
        "validation_metrics.json",
        "test_metrics.json",
        "distances.json"
    ]

    for c in tqdm(concepts, desc="Concepts (LR)", unit="concept", leave=True):
        table = normalize_table_name(c)

        for cmd_tpl in CONFIG["commands"]:
            if " -r l1" in cmd_tpl:
                reg_type = "l1"
            else:
                reg_type = "l2"

            fit_intercept = " -i" in cmd_tpl

            if "--mode b" in cmd_tpl:
                mode_dir = "binary_choice"
            else:
                mode_dir = "statement"

            method_dir = (
                f"{reg_type}_"
                f"{'fitted_intercept' if fit_intercept else 'zero_intercept'}"
            )

            out_dir = os.path.join(
                "vectors",
                model_dir,
                table,
                method_dir,
                mode_dir,
            )

            all_exist = True
            for filename in needed_files:
                full_path = os.path.join(out_dir, filename)
                if not os.path.exists(full_path):
                    all_exist = False
                    break

            if all_exist:
                continue

            cmd = (
                cmd_tpl
                .replace("$model", args.model)
                .replace("$c", c)
            )
            submit(cmd)


if __name__ == "__main__":
    main()

