import sqlite3
import torch
from transformers import LogitsProcessor, LogitsProcessorList
from injection_utils import inject
from helpers import normalize_table_name


INVENTORY_TEMPLATE = (
    'Given a statement: "You {stmt}."\n'
    "Which option best describes you?\n"
    "Options:\n"
    "(A). Very Accurate\n"
    "(B). Moderately Accurate\n"
    "(C). Neither Accurate Nor Inaccurate\n"
    "(D). Moderately Inaccurate\n"
    "(E). Very Inaccurate\n"
    "Answer:"
)
INVENTORY_SYSTEM = "You are being interviewed. Reply exclusively with A, B, C, D, or E. Do not ask anything."
SJT_SYSTEM = "You are being interviewed. Reply exclusively with one very short sentence in standard English. Do not ask anything."
ASSISTANT_PREFIX = "I would"
DIMENSION_SUFFIXES = [
    "_ocean_mention",
    "_cmni30_mention",
    "_cfni45_mention",
]
likert_map = {
    "A": 5,
    "B": 4,
    "C": 3,
    "D": 2,
    "E": 1,
}
likert_inverted_map = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
}


class ValidLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        super().__init__()
        self.allowed = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        idx = list(self.allowed)
        mask[:, idx] = scores[:, idx]
        return mask


def concept_to_dimension(name: str) -> str:
    base = name
    for suffix in DIMENSION_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    base = base.replace("_", " ")
    if not base:
        return base
    return base[0].upper() + base[1:]


def _normalize_concepts(concepts):
    if concepts is None:
        return None
    if isinstance(concepts, str):
        return [normalize_table_name(concepts)]
    result = []
    for c in concepts:
        result.append(normalize_table_name(c))
    return result


def load_inventory_rows(table_name, concepts=None):
    col = "dimension"
    with sqlite3.connect("../../data/psychometrics.db") as con:
        cur = con.cursor()
        if concepts:
            if isinstance(concepts, str):
                concepts_list = [concepts]
            else:
                concepts_list = list(concepts)
            dims = []
            for c in concepts_list:
                d = concept_to_dimension(c)
                if d not in dims:
                    dims.append(d)
            placeholders = ",".join("?" for _ in dims)
            sql = (
                f'SELECT lower({col}), item, key FROM "{table_name}" '
                f"WHERE {col} IN ({placeholders})"
            )
            cur.execute(sql, dims)
        else:
            sql = f'SELECT lower({col}), item, key FROM "{table_name}"'
            cur.execute(sql)
        rows = cur.fetchall()
    return rows


def load_sjts_rows(table_name, concepts):
    if concepts is None:
        return []
    if isinstance(concepts, str):
        concepts_list = [concepts]
    else:
        concepts_list = list(concepts)
    if not concepts_list:
        return []
    dims = []
    for c in concepts_list:
        d = concept_to_dimension(c)
        if d not in dims:
            dims.append(d)
    placeholders = ",".join("?" for _ in dims)
    col = "dimension"
    sql = (
        f'SELECT sjt FROM "{table_name}" '
        f"WHERE {col} IN ({placeholders})"
    )
    with sqlite3.connect("../../data/sjts.db") as con:
        cur = con.cursor()
        cur.execute(sql, dims)
        rows = cur.fetchall()
    sjts = []
    for row in rows:
        sjts.append(row[0])
    return sjts


def normalize_stmt_for_prompt(original_stmt):
    s = original_stmt.strip()
    if s.endswith("."):
        s = s[:-1]
    if s:
        s = s[0].lower() + s[1:]
    return s


def build_prompts(statements):
    prompts = []
    for stmt in statements:
        stmt_for_prompt = normalize_stmt_for_prompt(stmt)
        prompt = INVENTORY_TEMPLATE.format(stmt=stmt_for_prompt)
        prompts.append(prompt)
    return prompts


def prepare_logits_processor(tokenizer):
    allowed_ids = set()
    for ch in ["A", "B", "C", "D", "E"]:
        for variant in (ch, f" {ch}"):
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1:
                allowed_ids.add(toks[0])
    sorted_ids = sorted(allowed_ids)
    processor = ValidLogitsProcessor(sorted_ids)
    return LogitsProcessorList([processor])


def _score_inventory(statements, answers, keys):
    scores = []
    for i in range(len(statements)):
        ans = answers[i].strip()
        key = keys[i]
        if key == -1:
            score = likert_inverted_map[ans]
        else:
            score = likert_map[ans]
        scores.append(score)
    return statements, scores


def run_inventory(
    model,
    tokenizer,
    inventory,
    method,
    concepts,
    layers,
    model_name,
    fit_intercept,
    alphas,
    mode,
    batch_size,
    stride=None,
    evaluation_concepts=None,
    system_text=INVENTORY_SYSTEM,
    **generate_kwargs,
):
    if not evaluation_concepts:
        evaluation_concepts = concepts
    norm_concepts = _normalize_concepts(concepts)
    norm_evaluation_concepts = _normalize_concepts(evaluation_concepts)
    rows = load_inventory_rows(inventory, norm_evaluation_concepts)
    dimensions = []
    statements = []
    keys = []
    for dimension, item, key in rows:
        dimensions.append(dimension)
        statements.append(item)
        keys.append(key)
    prompts = build_prompts(statements)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    logits_proc = prepare_logits_processor(tokenizer)
    if model_name is None:
        model_name = getattr(model, "name_or_path", None)
    if model_name is None:
        raise ValueError("model_name must be provided or model.name_or_path must be set.")
    texts = inject(
        model=model,
        tokenizer=tokenizer,
        method=method,
        concepts=norm_concepts,
        prompts=prompts,
        layers=layers,
        model_name=model_name,
        alphas=alphas,
        assistant_prefix="",
        max_new_tokens=1,
        batch_size=batch_size,
        system_text=system_text,
        fit_intercept=fit_intercept,
        mode=mode,
        do_sample=False,
        logits_processor=logits_proc,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stride=stride,
        **generate_kwargs,
    )
    answers = []
    for text in texts:
        answers.append(text.strip())
    statements_scored, scores = _score_inventory(statements, answers, keys)
    return dimensions, statements_scored, scores


def run_sjts(
    model,
    tokenizer,
    inventory,
    method,
    concepts,
    layers,
    model_name,
    fit_intercept,
    alphas,
    mode,
    batch_size,
    stride=None,
    system_text=SJT_SYSTEM,
    assistant_prefix=ASSISTANT_PREFIX,
    **generate_kwargs,
):
    norm_concepts = _normalize_concepts(concepts)
    questions = load_sjts_rows(inventory, norm_concepts)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if model_name is None:
        model_name = getattr(model, "name_or_path", None)
    if model_name is None:
        raise ValueError("model_name must be provided or model.name_or_path must be set.")
    texts = inject(
        model=model,
        tokenizer=tokenizer,
        method=method,
        concepts=norm_concepts,
        system_text=system_text,
        prompts=questions,
        assistant_prefix=assistant_prefix,
        layers=layers,
        model_name=model_name,
        alphas=alphas,
        fit_intercept=fit_intercept,
        batch_size=batch_size,
        max_new_tokens=64,
        mode=mode,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stride=stride,
        **generate_kwargs,
    )
    answers = []
    for text in texts:
        answers.append(text.strip())
    return questions, answers
