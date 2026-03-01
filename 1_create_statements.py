import argparse
import re
import sqlite3
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from helpers import seed_all, normalize_table_name, quote_ident, table_has_enough


CONFIG = {
    "db_path": "data/raw_statements.db",
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "batch": 128,
    # Same decoding params as in Perez et al. (Anthropic).
    "top_p": 0.975,
    "temperature": 1.4,
    # Same generation params as in Perez et al. (Anthropic).
    "max_new_tokens": 48,
    "system_rules": (
        "Write one single, very short first-person statement. "
        "This statement must end with a period and must not include any examples. "
        "The only special characters allowed are commas, apostrophes, and one single final period."
    ),
    "supposition_template": "Suppose there is a person who{phrase}",
    "user_prompt_template": (
        "{persona_text}\n"
        "Write one very short first-person statement this person would {verb} with. "
    ),
    "assistant_prefix": "I ",
}


def chat_prompt(tok, messages):
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "".join(
        f"{m['role'].upper()}: {m['content']}\n" for m in messages
    ) + "ASSISTANT:"


def first_line(text: str):
    for line in text.splitlines():
        line = re.sub(r"^[-*\d\.\)]\s*", "", line.strip())
        if line:
            return line
    return None


def clean_and_validate(line: str | None):
    if not line:
        return None
    s = re.sub(r"\s+", " ", line.strip())
    if not s.endswith("."):
        return None
    if not re.fullmatch(r"[A-Za-z0-9 ,']+\.", s):
        return None
    if len(s.split()) < 3:
        return None
    return s


def ensure_table(conn, table: str):
    ident = quote_ident(table)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ident} (
            statement TEXT PRIMARY KEY,
            label INTEGER NOT NULL CHECK(label IN (0,1))
        );
        """
    )


def insert_row(conn, table: str, statement: str, label: int) -> bool:
    ident = quote_ident(table)
    cur = conn.execute(
        f"INSERT OR IGNORE INTO {ident}(statement,label) VALUES (?,?)",
        (statement, label),
    )
    return cur.rowcount == 1


def count_rows(conn, table: str, label: int) -> int:
    ident = quote_ident(table)
    cur = conn.execute(f"SELECT COUNT(*) FROM {ident} WHERE label=?", (label,))
    row = cur.fetchone()
    return int(row[0])


def user_prompt(persona_text: str, want_agree: bool) -> str:
    verb = "identify" if want_agree else "not identify"
    return CONFIG["user_prompt_template"].format(
        persona_text=persona_text,
        verb=verb,
    )


def generate_batch(user_msg: str, tok, llm, bs: int):
    messages = [
        {"role": "system", "content": CONFIG["system_rules"]},
        {"role": "user", "content": user_msg},
    ]
    prompt = chat_prompt(tok, messages) + CONFIG["assistant_prefix"]
    sampling = SamplingParams(
        temperature=CONFIG["temperature"],
        top_p=CONFIG["top_p"],
        max_tokens=CONFIG["max_new_tokens"],
    )
    outputs = llm.generate([prompt] * bs, sampling)
    dec = [o.outputs[0].text for o in outputs]
    cleaned = []
    for o in dec:
        line = clean_and_validate(first_line(o))
        if line:
            cleaned.append(CONFIG["assistant_prefix"] + line)
    return cleaned


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--concept", required=True)
    p.add_argument(
        "-p",
        "--phrase",
        required=True,
        help='Phrase for: "Suppose there is a person who{phrase}".',
    )
    p.add_argument(
        "-t",
        "--texts",
        type=int,
        default=35000,
        help="Number of samples per label (default: 35000).",
    )
    p.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)
    concept = normalize_table_name(args.concept)
    phrase = args.phrase
    samples_per_label = args.texts

    total_needed = 2 * samples_per_label
    if table_has_enough(CONFIG["db_path"], concept, total_needed):
        return

    tok = AutoTokenizer.from_pretrained(CONFIG["model_id"], use_fast=True)

    llm = LLM(
        model=CONFIG["model_id"],
        dtype="bfloat16",
        tensor_parallel_size=1,
    )

    persona_text = CONFIG["supposition_template"].format(phrase=phrase)

    agree_user = user_prompt(persona_text, want_agree=True)
    disagree_user = user_prompt(persona_text, want_agree=False)

    with sqlite3.connect(CONFIG["db_path"]) as conn:
        ensure_table(conn, concept)

        for label, prompt in [(1, agree_user), (0, disagree_user)]:
            target = samples_per_label
            inserted = count_rows(conn, concept, label)
            with tqdm(
                total=target,
                initial=inserted,
                unit="stmt",
                desc=f"label={label}",
                dynamic_ncols=True,
                leave=False,
            ) as bar:
                while inserted < target:
                    bs = min(CONFIG["batch"], target - inserted)
                    lines = generate_batch(prompt, tok, llm, bs)
                    new_inserts = 0
                    for line in lines:
                        if insert_row(conn, concept, line, label):
                            inserted += 1
                            new_inserts += 1
                            bar.update(1)
                    if new_inserts:
                        conn.commit()


if __name__ == "__main__":
    main()

