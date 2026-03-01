import argparse
import sqlite3
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from psychometric_utils import (
    INVENTORY_SYSTEM,
    SJT_SYSTEM,
    ASSISTANT_PREFIX,
    build_prompts,
    concept_to_dimension,
    load_inventory_rows,
    load_sjts_rows,
    prepare_logits_processor,
)
from sweeping_utils import CONFIG as SWEEPING_CONFIG, load_classifier
from data.inventory_to_constructs import inventory_to_dimensions
from data.personality_prompting import p2_descriptions, p2_descriptions_reversed
from helpers import (
    embed_batch,
    init_embed_model,
    embed_texts,
    init_fluency_model,
    fluency_scores
)


CONFIG = {
    "ocean_traits": ["Agreeableness", "Conscientiousness", "Extraversion", "Neuroticism", "Openness"],
    "out_root": Path("p2_results/"),
    "inventory_table": "mpi120",
    "sjts_table": "mpi120",
    "max_new_tokens_sjt": 64,
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("-bs", "--batch_size", type=int, default=128)
    ap.add_argument("-q", "--quantize", action="store_true")
    return ap.parse_args()


def short_model_name(model_id: str) -> str:
    s = model_id.strip()
    if "/" in s:
        s = s.split("/")[-1]
    return s.replace(" ", "_").replace(":", "_")


def trait_table(trait: str) -> str:
    return trait.strip().lower()


def trait_concept_name(trait: str) -> str:
    concepts = list(inventories_to_dimensions[CONFIG["inventory_table"]])
    for c in concepts:
        if concept_to_dimension(c) == trait:
            return c
    raise ValueError(
        f"No concept found for trait {trait!r} "
        f'in inventories_to_dimensions[{CONFIG["inventory_table"]!r}]'
    )


def init_sjt_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as con:
        cur = con.cursor()
        for trait in CONFIG["ocean_traits"]:
            t = trait_table(trait)
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS "{t}" (
                  class         INTEGER NOT NULL CHECK(class IN (0, 1)),
                  sjt           TEXT NOT NULL,
                  answer        TEXT NOT NULL,
                  concept_score REAL NOT NULL,
                  fluency_score REAL NOT NULL
                )
                """
            )
        con.commit()


def init_inventory_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as con:
        cur = con.cursor()
        for trait in CONFIG["ocean_traits"]:
            t = trait_table(trait)
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS "{t}" (
                  class INTEGER NOT NULL CHECK(class IN (0, 1)),
                  item  TEXT NOT NULL,
                  score REAL NOT NULL
                )
                """
            )
        con.commit()


def count_rows(con: sqlite3.Connection, table: str, cls_int: int) -> int:
    cur = con.cursor()
    cur.execute(f'SELECT COUNT(*) FROM "{table}" WHERE class=?', (int(cls_int),))
    return int(cur.fetchone()[0])


def already_done(sjt_con, inv_con, table: str, cls_int: int, need_sjt: int, need_inv: int) -> bool:
    return count_rows(sjt_con, table, cls_int) >= int(need_sjt) and count_rows(inv_con, table, cls_int) >= int(need_inv)


def build_inputs(tokenizer, system_text: str, user_text: str, assistant_prefix: str):
    messages = [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        enc = tokenizer(prompt_text, add_special_tokens=False)
        ids = enc["input_ids"]
    else:
        prompt_text = system_text + "\n\n" + user_text + "\n\nAssistant:"
        enc = tokenizer(prompt_text, add_special_tokens=True)
        ids = enc["input_ids"]
    if assistant_prefix:
        ids = ids + tokenizer(assistant_prefix, add_special_tokens=False)["input_ids"]
    return torch.tensor(ids, dtype=torch.long)


@torch.no_grad()
def generate_texts(model, tokenizer, system_text: str, prompts, assistant_prefix: str, max_new_tokens: int, batch_size: int, logits_processor=None):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    texts = []
    i = 0
    while i < len(prompts):
        batch_prompts = prompts[i : i + batch_size]
        ids_list = []
        for p in batch_prompts:
            ids_list.append(build_inputs(tokenizer, system_text, p, assistant_prefix))
        batch = tokenizer.pad({"input_ids": ids_list}, return_tensors="pt")
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        base_len = input_ids.shape[1]
        b = 0
        while b < out.size(0):
            gen_ids = out[b, base_len:]
            tail = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            texts.append(((assistant_prefix + " " + tail).strip()) if assistant_prefix else tail)
            b += 1
        i += batch_size
    return texts


def clamp_answer_letter(x: str) -> str:
    s = x.strip()
    if not s:
        return "C"
    ch = s[0].upper()
    if ch in ["A", "B", "C", "D", "E"]:
        return ch
    if len(s) >= 2 and s[0] == " " and s[1].upper() in ["A", "B", "C", "D", "E"]:
        return s[1].upper()
    return "C"


def likert_score(letter: str, key: int) -> int:
    if key == -1:
        if letter == "A":
            return 1
        if letter == "B":
            return 2
        if letter == "C":
            return 3
        if letter == "D":
            return 4
        return 5
    if letter == "A":
        return 5
    if letter == "B":
        return 4
    if letter == "C":
        return 3
    if letter == "D":
        return 2
    return 1


def main():
    args = parse_args()

    out_dir = CONFIG["out_root"] / short_model_name(args.model)
    sjt_db_path = out_dir / "sjts_responses.db"
    inv_db_path = out_dir / "inventory_responses.db"

    init_sjt_db(sjt_db_path)
    init_inventory_db(inv_db_path)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant_cfg if args.quantize else None,
        low_cpu_mem_usage=True,
    ).eval()

    embed_tok, embed_model = init_embed_model()

    fluency_tok, fluency_model = init_fluency_model()
    inv_logits_proc = prepare_logits_processor(tok)

    with sqlite3.connect(str(sjt_db_path)) as sjt_con, sqlite3.connect(str(inv_db_path)) as inv_con:
        for trait in CONFIG["ocean_traits"]:
            tname = trait_table(trait)
            concept_name = trait_concept_name(trait)
            probe_clf = load_classifier(concept_name)

            inv_rows = load_inventory_rows(CONFIG["inventory_table"], concepts=[concept_name])
            inv_items = []
            inv_keys = []
            for item, key in inv_rows:
                inv_items.append(item)
                inv_keys.append(int(key))

            questions = load_sjts_rows(CONFIG["sjts_table"], concepts=[concept_name])

            need_inv = len(inv_items)
            need_sjt = len(questions)

            for cls_int in [0, 1]:
                if already_done(sjt_con, inv_con, tname, cls_int, need_sjt, need_inv):
                    continue

                persona = p2_descriptions[trait] if cls_int == 1 else p2_descriptions_reversed[trait]
                inv_system = persona + "\n" + INVENTORY_SYSTEM
                sjt_system = persona + "\n" + SJT_SYSTEM

                inv_prompts = build_prompts(inv_items)
                inv_texts = generate_texts(
                    model=model,
                    tokenizer=tok,
                    system_text=inv_system,
                    prompts=inv_prompts,
                    assistant_prefix="",
                    max_new_tokens=1,
                    batch_size=int(args.batch_size),
                    logits_processor=inv_logits_proc,
                )

                inv_payload = []
                i = 0
                while i < len(inv_items):
                    letter = clamp_answer_letter(inv_texts[i])
                    score = float(likert_score(letter, inv_keys[i]))
                    inv_payload.append((int(cls_int), inv_items[i], score))
                    i += 1

                inv_con.cursor().executemany(
                    f'INSERT INTO "{tname}" (class, item, score) VALUES (?, ?, ?)',
                    inv_payload,
                )
                inv_con.commit()

                sjt_texts = generate_texts(
                    model=model,
                    tokenizer=tok,
                    system_text=sjt_system,
                    prompts=questions,
                    assistant_prefix=ASSISTANT_PREFIX,
                    max_new_tokens=CONFIG["max_new_tokens_sjt"],
                    batch_size=int(args.batch_size),
                    logits_processor=None,
                )

                flu_scores = fluency_scores(
                    fluency_tok,
                    fluency_model,
                    [s.strip() for s in sjt_texts],
                    batch_size=int(SWEEPING_CONFIG["fluency_batch_size"]),
                )
                X = embed_texts(
                    embed_tok,
                    embed_model,
                    sjt_texts,
                    int(SWEEPING_CONFIG["probe_embed_batch"]),
                )

                probs = probe_clf.predict_proba(X)
                classes = list(probe_clf.classes_)
                pos_index = classes.index(1)

                sjt_payload = []
                j = 0
                while j < len(questions):
                    sjt_payload.append((int(cls_int), questions[j], sjt_texts[j], float(probs[j][pos_index]), float(flu_scores[j])))
                    j += 1

                sjt_con.cursor().executemany(
                    f'INSERT INTO "{tname}" (class, sjt, answer, concept_score, fluency_score) VALUES (?, ?, ?, ?, ?)',
                    sjt_payload,
                )
                sjt_con.commit()

    print(f"Wrote: {sjt_db_path}")
    print(f"Wrote: {inv_db_path}")


if __name__ == "__main__":
    main()
