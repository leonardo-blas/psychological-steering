import re
import sqlite3
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
FLUENCY_MODEL_ID = "cointegrated/roberta-large-cola-krishna2020"


def seed_all(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def init_embed_model():
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL_ID, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(
        EMBED_MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    model.to("cuda")
    model.eval()
    return tok, model


@torch.no_grad()
def embed_batch(embed_tok, embed_model, texts):
    enc = embed_tok(
        texts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    enc = enc.to(embed_model.device)
    out = embed_model(**enc)
    x = out.last_hidden_state[:, -1]
    x = F.normalize(x, p=2, dim=1)
    return x


@torch.no_grad()
def embed_texts(embed_tok, embed_model, texts, batch_size: int):
    all_embs = []
    bs = int(batch_size)
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        emb = embed_batch(embed_tok, embed_model, batch)
        all_embs.append(emb)
    embs = torch.cat(all_embs, dim=0)
    return embs.to(torch.float32).cpu().numpy()


def init_fluency_model():
    model_id = FLUENCY_MODEL_ID
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to("cuda")
    model.eval()
    return tok, model


@torch.no_grad()
def fluency_filter_batch(flu_tok, flu_model, texts, threshold: float):
    enc = flu_tok(
        texts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    enc = enc.to(flu_model.device)
    out = flu_model(**enc)
    probs = out.logits.softmax(dim=1)
    return (probs[:, 0] >= float(threshold)).tolist()


@torch.no_grad()
def fluency_scores(fluency_tok, fluency_model, texts, batch_size: int = 512):
    if not texts:
        return []
    bs = int(batch_size)
    if bs <= 0:
        raise ValueError("batch_size must be > 0.")
    scores = []
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        enc = fluency_tok(batch, padding=True, truncation=True, return_tensors="pt")
        for k in enc:
            enc[k] = enc[k].to(fluency_model.device)
        out = fluency_model(**enc)
        probs = out.logits.softmax(1)[:, 0]
        scores.extend(probs.detach().cpu().tolist())
    return scores


def normalize_table_name(s: str) -> str:
    t = s.strip().lower()
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"[^a-z0-9_]", "_", t)
    t = re.sub(r"_+", "_", t)
    t = t.strip("_")
    return t


def table_has_enough(db_path: str, table: str, total_needed: int) -> bool:
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file does not exist: {db_path}")
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        if not cur.fetchone():
            n = 0
        else:
            cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
            row = cur.fetchone()
            n =  0 if not row else row[0]
    return n >= total_needed
