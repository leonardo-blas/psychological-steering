import re
import sqlite3
import random
import numpy as np
import torch


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


def seed_all(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def normalize_table_name(s: str) -> str:
    t = s.strip().lower()
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"[^a-z0-9_]", "_", t)
    t = re.sub(r"_+", "_", t)
    t = t.strip("_")
    return t


def quote_ident(name: str) -> str:
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    row = cur.fetchone()
    return row is not None


def get_table_rowcount_conn(conn: sqlite3.Connection, table: str) -> int:
    if not table_exists(conn, table):
        return 0
    ident = quote_ident(table)
    cur = conn.execute(f"SELECT COUNT(*) FROM {ident}")
    row = cur.fetchone()
    if row is None:
        return 0
    return int(row[0])


def get_table_rowcount(db_path: str, table: str) -> int:
    with sqlite3.connect(db_path) as conn:
        return get_table_rowcount_conn(conn, table)


def table_has_enough(db_path: str, table: str, total_needed: int) -> bool:
    n = get_table_rowcount(db_path, table)
    return n >= total_needed


def format_atomic10x_head(text: str) -> str:
    s = text
    name_x = "Alex" if "Alex" not in s else "Avery"
    name_y = "Brook" if "Brook" not in s else "Blake"
    name_z = "Charlie" if "Charlie" not in s else "Cameron"
    s = re.sub(r"\b[Pp]ersonX\b", name_x, s)
    s = re.sub(r"\b[Pp]ersonY\b", name_y, s)
    s = re.sub(r"\b[Pp]ersonZ\b", name_z, s)
    return s if s.endswith(".") else s + "."


def wrap_title(title: str, max_len: int = 40) -> str:
    if len(title) <= max_len:
        return title
    words = title.split()
    lines = []
    current = ""
    for w in words:
        if not current:
            current = w
        elif len(current) + 1 + len(w) <= max_len:
            current = current + " " + w
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    return "\n".join(lines)

