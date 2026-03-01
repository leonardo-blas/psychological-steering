import os
import json
import csv
import sqlite3
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
from helpers import seed_all


CONFIG = {
    "seed": 42,
    "sjts_db_path": "data/sjts.db",
    "embed_model_id": "Qwen/Qwen3-Embedding-0.6B",
    "batch_size": 1024,
    "big5_dimensions": [
        "Openness",
        "Conscientiousness",
        "Extraversion",
        "Agreeableness",
        "Neuroticism",
    ],
    "dark_triad_dimensions": [
        "Machiavellianism",
        "Narcissism",
        "Psychopathy",
    ],
    "hexaco_dimensions": [
        "Honesty-Humility",
        "Emotionality",
        "Extraversion",
        "Agreeableness",
        "Conscientiousness",
        "Openness to Experience",
    ],
    "trait_json_path": "data/TRAIT.json",
    "zhang_tsv_path": "data/sjts_zhang.tsv",
    "oostrom_tsv_path": "data/sjts_oostrom.tsv",
    "clifford_tsv_path": "data/sjts_clifford.tsv",
    "mpi_table": "mpi120",
    "hexaco_table": "hexaco60",
    "sd3_table": "sd3",
    "mfq30_table": "mfq30",
    "out_dir": "sjts_validation",
    "grid_alpha": 0.10,
    "grid_color": "0.35",
    "grid_lw": 0.8,
    "baby_gray": "#CED4DA",
    "baby_orange": "#FFB84D"
}
mpl.rcParams["figure.dpi"] = 600


def display_name(source: str) -> str:
    if source == "TRAIT":
        return "TRAIT"
    if source == "IPIP":
        return "Our SJTs (IPIP-120 based)"
    if source == "HEXACO":
        return "Our SJTs (HEXACO-60 based)"
    if source == "SD3":
        return "Our SJTs (SD3 based)"
    if source == "Zhang":
        return "Zhang et al."
    if source == "Oostrom":
        return "Oostrom et al."
    if source == "Clifford":
        return "Clifford et al."
    if source == "MFQ30":
        return "Our SJTs (MFQ-30 based)"
    return source.capitalize()


def normalize_big5_dim(name: str | None) -> str | None:
    if name is None:
        return None
    key = name.strip().lower()
    key = key.replace("_", " ")
    key = key.replace("-", " ")
    key = " ".join(key.split())
    if "open" in key or key == "o":
        return "Openness"
    if "conscient" in key or key == "c":
        return "Conscientiousness"
    if "extrav" in key or "extro" in key or key == "e":
        return "Extraversion"
    if "agree" in key or key == "a":
        return "Agreeableness"
    if "neuro" in key or "emotional stability" in key or key == "n":
        return "Neuroticism"
    return None


def normalize_hexaco_dim(name: str | None) -> str | None:
    if name is None:
        return None
    key = name.strip().lower()
    key = key.replace("_", " ")
    key = key.replace("-", " ")
    key = " ".join(key.split())
    if "honesty" in key or "humility" in key or key == "h" or key.startswith("h "):
        return "Honesty-Humility"
    if "emotional" in key or key == "e" or key.startswith("e "):
        return "Emotionality"
    if "extrav" in key or key == "x" or key.startswith("x "):
        return "Extraversion"
    if "agree" in key or key == "a" or key.startswith("a "):
        return "Agreeableness"
    if "conscient" in key or key == "c" or key.startswith("c "):
        return "Conscientiousness"
    if "open" in key or key == "o" or key.startswith("o "):
        return "Openness to Experience"
    return None


def normalize_dark_triad_dim(name: str | None) -> str | None:
    if name is None:
        return None
    key = name.strip().lower()
    key = key.replace("_", " ")
    key = key.replace("-", " ")
    key = " ".join(key.split())
    if "machiavell" in key or key == "m":
        return "Machiavellianism"
    if "narciss" in key or key == "n":
        return "Narcissism"
    if "psychopath" in key or key == "p":
        return "Psychopathy"
    return None


def load_trait_texts():
    texts = []
    sources = []
    dims = []
    with open(CONFIG["trait_json_path"], "r", encoding="utf-8") as f:
        data = json.load(f)
    for row in data:
        personality_raw = row.get("personality")
        dim_norm = normalize_big5_dim(personality_raw)
        if dim_norm is None:
            dim_norm = normalize_dark_triad_dim(personality_raw)
        if dim_norm is None:
            continue
        if dim_norm not in CONFIG["big5_dimensions"] and dim_norm not in CONFIG["dark_triad_dimensions"]:
            continue
        situation = row.get("situation", "")
        query = row.get("query", "")
        combined = (situation + " " + query).strip()
        if not combined:
            continue
        texts.append(combined)
        sources.append("TRAIT")
        dims.append(dim_norm)
    return texts, sources, dims


def load_mpi_sjt_texts():
    texts = []
    sources = []
    dims = []
    with sqlite3.connect(CONFIG["sjts_db_path"]) as conn:
        table = CONFIG["mpi_table"]
        cur = conn.cursor()
        cur.execute(f"SELECT dimension, sjt FROM {table}")
        rows = cur.fetchall()
    for dim, sjt in rows:
        if dim is None or sjt is None:
            continue
        dim_norm = normalize_big5_dim(dim)
        if dim_norm is None:
            continue
        if dim_norm not in CONFIG["big5_dimensions"]:
            continue
        sjt_text = sjt.strip()
        if not sjt_text:
            continue
        texts.append(sjt_text)
        sources.append("IPIP")
        dims.append(dim_norm)
    return texts, sources, dims


def load_hexaco_sjt_texts():
    texts = []
    sources = []
    dims = []
    with sqlite3.connect(CONFIG["sjts_db_path"]) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT dimension, sjt FROM {CONFIG['hexaco_table']}")
        rows = cur.fetchall()
    for dim, sjt in rows:
        if dim is None or sjt is None:
            continue
        dim_norm = normalize_hexaco_dim(dim)
        if dim_norm is None:
            continue
        if dim_norm not in CONFIG["hexaco_dimensions"]:
            continue
        sjt_text = sjt.strip()
        if not sjt_text:
            continue
        texts.append(sjt_text)
        sources.append("HEXACO")
        dims.append(dim_norm)
    return texts, sources, dims


def load_sd3_sjt_texts():
    texts = []
    sources = []
    dims = []
    with sqlite3.connect(CONFIG["sjts_db_path"]) as conn:
        table = CONFIG["sd3_table"]
        cur = conn.cursor()
        cur.execute(f"SELECT dimension, sjt FROM {table}")
        rows = cur.fetchall()
    for dim, sjt in rows:
        if dim is None or sjt is None:
            continue
        dim_norm = normalize_dark_triad_dim(dim)
        if dim_norm is None:
            continue
        if dim_norm not in CONFIG["dark_triad_dimensions"]:
            continue
        sjt_text = sjt.strip()
        if not sjt_text:
            continue
        texts.append(sjt_text)
        sources.append("SD3")
        dims.append(dim_norm)
    return texts, sources, dims


def load_zhang_texts():
    texts = []
    sources = []
    dims = []
    path = CONFIG["zhang_tsv_path"]
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            llm = (row.get("LLM") or "").strip()
            if llm != "gpt-4o":
                continue
            cot_raw = (row.get("COT") or "").strip()
            if cot_raw == "":
                continue
            try:
                cot_val = float(cot_raw)
            except ValueError:
                continue
            if cot_val != 0.0:
                continue
            stem = (row.get("stem") or "").strip()
            if not stem:
                continue
            dim_raw = (row.get("construct") or "").strip()
            dim_norm = normalize_hexaco_dim(dim_raw)
            if dim_norm is None:
                continue
            if dim_norm not in CONFIG["hexaco_dimensions"]:
                continue
            texts.append(stem)
            sources.append("Zhang")
            dims.append(dim_norm)
    return texts, sources, dims


def load_oostrom_texts():
    texts = []
    sources = []
    dims = []
    path = CONFIG["oostrom_tsv_path"]
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            dim_raw = (row.get("Dimension") or "").strip()
            sjt = (row.get("SJT") or "").strip()
            if not sjt:
                continue
            dim_norm = normalize_hexaco_dim(dim_raw)
            if dim_norm is None:
                continue
            if dim_norm not in CONFIG["hexaco_dimensions"]:
                continue
            texts.append(sjt)
            sources.append("Oostrom")
            dims.append(dim_norm)
    return texts, sources, dims


def load_clifford_texts():
    path = CONFIG["clifford_tsv_path"]
    texts = []
    sources = []
    dims = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                continue
            dim = row[0]
            sjt = row[1]
            if dim is None or sjt is None:
                continue
            dim = dim.strip()
            sjt = sjt.strip()
            if not dim or not sjt:
                continue
            texts.append(sjt)
            sources.append("Clifford")
            dims.append(dim)
    return texts, sources, dims


def load_mfq30_sjt_texts():
    texts = []
    sources = []
    dims = []
    with sqlite3.connect(CONFIG["sjts_db_path"]) as conn:
        table = CONFIG["mfq30_table"]
        cur = conn.cursor()
        cur.execute(f"SELECT dimension, sjt FROM {table}")
        rows = cur.fetchall()
    for dim, sjt in rows:
        if dim is None or sjt is None:
            continue
        dim = dim.strip()
        sjt = sjt.strip()
        if not dim or not sjt:
            continue
        texts.append(sjt)
        sources.append("MFQ30")
        dims.append(dim)
    return texts, sources, dims


@torch.no_grad()
def embed_batch(embed_tok, embed_model, texts):
    x = embed_tok(
        texts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    x = x.to(embed_model.device)
    x = embed_model(**x)
    x = x.last_hidden_state[:, -1]
    return F.normalize(x, p=2, dim=1)


def embed_texts(texts):
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["embed_model_id"], padding_side="left")
    model = AutoModel.from_pretrained(CONFIG["embed_model_id"], dtype=torch.bfloat16).to("cuda")
    model.eval()
    all_embeddings = []
    start = 0
    while start < len(texts):
        end = min(start + CONFIG["batch_size"], len(texts))
        emb = embed_batch(tokenizer, model, texts[start:end])
        all_embeddings.append(emb.to(dtype=torch.float32).cpu().numpy())
        start = end
    return np.concatenate(all_embeddings, axis=0)


def plot_pca(X, src_labels, pdf_path):
    if X.shape[0] == 0:
        return
    pca = PCA(n_components=2, random_state=CONFIG["seed"])
    coords = pca.fit_transform(X)
    xs = coords[:, 0]
    ys = coords[:, 1]

    sources_unique = []
    for s in src_labels:
        if s not in sources_unique:
            sources_unique.append(s)

    src_to_color = {}
    for s in sources_unique:
        if s in ("IPIP", "HEXACO", "SD3"):
            src_to_color[s] = CONFIG["baby_orange"]
        else:
            src_to_color[s] = CONFIG["baby_gray"]

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(2.75, 1.75))
        ax.set_axisbelow(True)
        ax.grid(True, which="major", alpha=CONFIG["grid_alpha"], color=CONFIG["grid_color"], linewidth=CONFIG["grid_lw"])

        for s in sources_unique:
            xs_s = []
            ys_s = []
            for i in range(len(xs)):
                if src_labels[i] == s:
                    xs_s.append(xs[i])
                    ys_s.append(ys[i])
            if not xs_s:
                continue
            ax.scatter(
                xs_s,
                ys_s,
                s=8,
                alpha=0.7,
                color=src_to_color[s],
                edgecolors="none",
                label=display_name(s),
            )

        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)

        if sources_unique:
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=len(sources_unique),
                frameon=True,
                fontsize=8,
                markerscale=1.2,
                handletextpad=0.4,
                columnspacing=0.9,
                borderaxespad=0.0,
            )

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def centroid_similarity(emb_a, emb_b):
    if len(emb_a) == 0 or len(emb_b) == 0:
        return None
    centroid_a = emb_a.mean(axis=0)
    centroid_b = emb_b.mean(axis=0)
    return 1 - cosine(centroid_a, centroid_b)


def compare_by_dim(emb, sources, dims, left_src, right_src, prefix, label, dim_values):
    for d in dim_values:
        idx_l = [i for i, (s, dim) in enumerate(zip(sources, dims)) if s == left_src and dim == d]
        idx_r = [i for i, (s, dim) in enumerate(zip(sources, dims)) if s == right_src and dim == d]
        if not idx_l or not idx_r:
            continue

        sim = centroid_similarity(emb[idx_l], emb[idx_r])
        X = np.vstack([emb[idx_l], emb[idx_r]])
        src_labels = [left_src] * len(idx_l) + [right_src] * len(idx_r)
        dim_slug = d.lower().replace(" ", "_").replace("-", "_")
        pdf = os.path.join(CONFIG["out_dir"], f"{prefix}_{dim_slug}.pdf")
        plot_pca(X, src_labels, pdf)
        print(f"{label} - {d}: {sim:.6f}")


def main():
    seed_all(CONFIG["seed"])
    os.makedirs(CONFIG["out_dir"], exist_ok=True)

    trait_texts, trait_sources, trait_dims = load_trait_texts()
    mpi_texts, mpi_sources, mpi_dims = load_mpi_sjt_texts()
    hex_texts, hex_sources, hex_dims = load_hexaco_sjt_texts()
    sd3_texts, sd3_sources, sd3_dims = load_sd3_sjt_texts()
    zhang_texts, zhang_sources, zhang_dims = load_zhang_texts()
    oost_texts, oost_sources, oost_dims = load_oostrom_texts()
    cliff_texts, cliff_sources, cliff_dims = load_clifford_texts()
    mfq_texts, mfq_sources, mfq_dims = load_mfq30_sjt_texts()

    datasets = [
        (trait_texts, trait_sources, trait_dims),
        (mpi_texts, mpi_sources, mpi_dims),
        (hex_texts, hex_sources, hex_dims),
        (sd3_texts, sd3_sources, sd3_dims),
        (zhang_texts, zhang_sources, zhang_dims),
        (oost_texts, oost_sources, oost_dims),
        (cliff_texts, cliff_sources, cliff_dims),
        (mfq_texts, mfq_sources, mfq_dims),
    ]
    texts_all = []
    sources_all = []
    dims_all = []
    for texts, sources, dims in datasets:
        texts_all.extend(texts)
        sources_all.extend(sources)
        dims_all.extend(dims)

    emb = embed_texts(texts_all)

    print("\n=== TRAIT vs Ours (IPIP-120 based) ===")
    compare_by_dim(
        emb,
        sources_all,
        dims_all,
        "TRAIT",
        "IPIP",
        "trait",
        "TRAIT vs Ours (IPIP-120 based)",
        CONFIG["big5_dimensions"],
    )

    print("\n=== TRAIT vs Ours (SD3 based) ===")
    compare_by_dim(
        emb,
        sources_all,
        dims_all,
        "TRAIT",
        "SD3",
        "trait_sd3",
        "TRAIT vs Ours (SD3 based)",
        CONFIG["dark_triad_dimensions"],
    )

    print("\n=== Oostrom vs Ours (HEXACO-60 based) ===")
    compare_by_dim(
        emb,
        sources_all,
        dims_all,
        "Oostrom",
        "HEXACO",
        "oostrom",
        "Oostrom vs Ours (HEXACO-60 based)",
        CONFIG["hexaco_dimensions"],
    )

    print("\n=== Zhang vs Ours (HEXACO-60 based) ===")
    compare_by_dim(
        emb,
        sources_all,
        dims_all,
        "Zhang",
        "HEXACO",
        "zhang",
        "Zhang vs Ours (HEXACO-60 based)",
        CONFIG["hexaco_dimensions"],
    )

    print("\n=== Clifford vs Ours (MFQ-30 based) ===")
    compare_by_dim(
        emb,
        sources_all,
        dims_all,
        "Clifford",
        "MFQ30",
        "clifford",
        "Clifford vs Ours (MFQ-30 based)",
        sorted({d for d in dims_all if d is not None}),
    )


if __name__ == "__main__":
    main()

