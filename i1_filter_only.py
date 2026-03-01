#!/usr/bin/env python

import sqlite3
import subprocess
import sys
from tqdm.auto import tqdm

CONFIG = {
    "concepts_db": "data/concepts.db",
    "concepts_table": "concepts",
    "filter_script": "2_filter_statements.py",
}


def get_concepts():
    conn = sqlite3.connect(CONFIG["concepts_db"])
    cur = conn.cursor()
    cur.execute(
        f"SELECT concept, phrase FROM {CONFIG['concepts_table']}"
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def main():
    rows = get_concepts()
    for concept, phrase in tqdm(
        rows,
        desc="filtering concepts",
        leave=True,
    ):
        cmd = [
            sys.executable,
            CONFIG["filter_script"],
            "-c",
            concept,
            "-p",
            phrase,
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(
                f"\n[ERROR] concept={concept!r} phrase={phrase!r} failed: {e}\n",
                file=sys.stderr,
            )
            continue


if __name__ == "__main__":
    main()

