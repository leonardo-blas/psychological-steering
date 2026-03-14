import argparse
import os
import sqlite3
import subprocess
from tqdm.auto import tqdm
from helpers import normalize_table_name


CONFIG = {
    "concepts_db": "data/concepts.db",
    "commands": [
        'python3 -u 4_create_vectors_meandiff.py --mode b --model "$model" --concept "$c"',
        'python3 -u 4_create_vectors_meandiff.py --mode s --model "$model" --concept "$c"',
    ],
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    return ap.parse_args()


def load_concepts(db_path: str):
    concepts = []
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT concept FROM vector_concepts")
        for (c,) in cur.fetchall():
            c = str(c).strip()
            if c:
                concepts.append(c)
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
        "distances.json"
    ]

    for c in tqdm(concepts, desc="Concepts (meandiff)", unit="concept", leave=True):
        table = normalize_table_name(c)

        for cmd_tpl in CONFIG["commands"]:
            if "--mode b" in cmd_tpl:
                mode_dir = "binary_choice"
            else:
                mode_dir = "statement"

            out_dir = os.path.join(
                "vectors",
                model_dir,
                table,
                "meandiff",
                mode_dir,
            )

            all_exist = True
            for filename in needed_files:
                full_path = os.path.join(out_dir, filename)
                if not os.path.exists(full_path):
                    all_exist = False
                    break

            if all_exist:
                pass
                #continue

            cmd = (
                cmd_tpl
                .replace("$model", args.model)
                .replace("$c", c)
            )
            submit(cmd)


if __name__ == "__main__":
    main()

