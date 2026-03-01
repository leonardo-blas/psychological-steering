import os
import json
import glob
import sqlite3
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl


CONFIG = {
    "seed": 42,
    "anthropic_dir": "data/anthropic_statements",
    "statements_db_path": "data/statements.db",
    #"statements_db_path": "../injections/data/best_follows_framework/statements.db",
    "out_dir": "statements_validation",
    "embed_model_id": "Qwen/Qwen3-Embedding-0.6B",
    "batch_size": 512,
    "concepts": ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", "psychopathy", "narcissism", "machiavellianism"],
    "trait_acronym":  {
        "openness": "O",
        "conscientiousness": "C",
        "extraversion": "E",
        "agreeableness": "A",
        "neuroticism": "N",
        "psychopathy": "P",
        "narcissism": "N",
        "machiavellianism": "M"
    },
    "grid_alpha":  0.10,
    "grid_color":  "0.35",
    "grid_lw":  0.8,
    "anthropic_gray":  "#CED4DA",
    "neg_teal":  "#6D9F71",
    "pos_coral":  "#EA9AB2",
}
mpl.rcParams["figure.dpi"] = 600


def trait_acro(trait: str) -> str:
    return CONFIG["trait_acronym"].get((trait or "").lower(), "?")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,))
    return cur.fetchone() is not None


def display_name(key: str) -> str:
    if key == "PEREZ":
        return "Perez et al."
    try:
        acro, lab = key.split("_", 1)
        arrow = r"\uparrow" if int(lab) == 1 else r"\downarrow"
        return rf"$\mathrm{{{acro}}}^{{{arrow}}}$"
    except Exception:
        return key


def load_anthropic_statements():
    texts, sources, traits, labels = [], [], [], []
    for fp in sorted(glob.glob(os.path.join(CONFIG["anthropic_dir"], "*.jsonl"))):
        trait = os.path.splitext(os.path.basename(fp))[0].strip().lower()
        if trait not in CONFIG["concepts"]:
            continue
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                s = (row.get("statement") or "").strip()
                if not s:
                    continue
                texts.append(s)
                sources.append("ANTHROPIC")
                traits.append(trait)
                labels.append(None)
    return texts, sources, traits, labels


def load_our_statements():
    texts, sources, traits, labels = [], [], [], []
    with sqlite3.connect(CONFIG["statements_db_path"]) as conn:
        for t in CONFIG["concepts"]:
            if not table_exists(conn, t):
                continue
            cur = conn.cursor()
            cur.execute(f'SELECT statement, label FROM "{t}"')
            for stmt, lab in cur.fetchall():
                if stmt is None:
                    continue
                s = stmt.strip()
                if not s:
                    continue
                if lab not in (0, 1):
                    continue
                texts.append(s)
                sources.append("OURS")
                traits.append(t)
                labels.append(int(lab))
    return texts, sources, traits, labels


def embed_texts(texts):
    if len(texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["embed_model_id"])
    model = AutoModel.from_pretrained(CONFIG["embed_model_id"], dtype=torch.bfloat16).to("cuda")
    model.eval()

    all_emb = []
    with torch.no_grad():
        start = 0
        while start < len(texts):
            end = min(start + CONFIG["batch_size"], len(texts))
            batch = [texts[i] for i in range(start, end)]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to("cuda")
            out = model(**enc)
            pooled = out.last_hidden_state.mean(dim=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1).to(dtype=torch.float32)
            all_emb.append(pooled.cpu().numpy())
            start = end

    return np.concatenate(all_emb, axis=0)


def point_style(src: str, trait: str, lab: int | None):
    if src == "ANTHROPIC":
        return "PEREZ", CONFIG["anthropic_gray"]
    acro = trait_acro(trait)
    if lab == 0:
        return f"{acro}_0", CONFIG["neg_teal"]
    if lab == 1:
        return f"{acro}_1", CONFIG["pos_coral"]
    return f"{acro}_?", CONFIG["pos_coral"]


def plot_pca(X, sources, traits, labels, pdf_path):
    if X.shape[0] == 0:
        return

    pca = PCA(n_components=2, random_state=CONFIG["seed"])
    coords = pca.fit_transform(X)
    xs = coords[:, 0]
    ys = coords[:, 1]

    group_keys = []
    group_xs = {}
    group_ys = {}
    group_colors = {}

    for i in range(len(xs)):
        key, col = point_style(sources[i], traits[i], labels[i])
        if key not in group_xs:
            group_xs[key] = []
            group_ys[key] = []
            group_colors[key] = col
            group_keys.append(key)
        group_xs[key].append(xs[i])
        group_ys[key].append(ys[i])

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(2.75, 1.75))

        ax.set_axisbelow(True)
        ax.grid(True, which="major", alpha=CONFIG["grid_alpha"], color=CONFIG["grid_color"], linewidth=CONFIG["grid_lw"])

        for key in group_keys:
            ax.scatter(
                group_xs[key],
                group_ys[key],
                s=8,
                alpha=0.7,
                color=group_colors[key],
                edgecolors="none",
                label=display_name(key),
            )

        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)

        if group_keys:
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=len(group_keys),
                frameon=True,
                fontsize=8,
                markerscale=1.2,
                handletextpad=0.4,
                columnspacing=0.9,
                borderaxespad=0.0,
            )

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def centroid_similarity(emb_a, emb_b):
    if len(emb_a) == 0 or len(emb_b) == 0:
        return None
    ca = emb_a.mean(axis=0)
    cb = emb_b.mean(axis=0)
    return 1 - cosine(ca, cb)


def report_sims_for_subset(emb, sources, labels, title: str):
    idx_a = [i for i, s in enumerate(sources) if s == "ANTHROPIC"]
    idx_0 = [i for i, (s, lab) in enumerate(zip(sources, labels)) if s == "OURS" and lab == 0]
    idx_1 = [i for i, (s, lab) in enumerate(zip(sources, labels)) if s == "OURS" and lab == 1]

    if idx_a and idx_0:
        print(f"{title}  Perez et al. vs (label 0): {centroid_similarity(emb[idx_a], emb[idx_0]):.6f}")
    else:
        print(f"{title}  Perez et al. vs (label 0): skipped")

    if idx_a and idx_1:
        print(f"{title}  Perez et al. vs (label 1): {centroid_similarity(emb[idx_a], emb[idx_1]):.6f}")
    else:
        print(f"{title}  Perez et al. vs (label 1): skipped")


def compare_per_trait(emb, sources, traits, labels, out_dir):
    for t in CONFIG["concepts"]:
        idx = [i for i, tr in enumerate(traits) if tr == t]
        if not idx:
            continue
        X = emb[idx]
        src_plot = [sources[i] for i in idx]
        tr_plot = [traits[i] for i in idx]
        lab_plot = [labels[i] for i in idx]

        report_sims_for_subset(X, src_plot, lab_plot, f"{t}:")
        plot_pca(X, src_plot, tr_plot, lab_plot, os.path.join(out_dir, f"anthropic_vs_ours_{t}.pdf"))


def main():
    seed_all(CONFIG["seed"])
    os.makedirs(CONFIG["out_dir"], exist_ok=True)

    a_texts, a_sources, a_traits, a_labels = load_anthropic_statements()
    o_texts, o_sources, o_traits, o_labels = load_our_statements()

    texts_all = a_texts + o_texts
    sources_all = a_sources + o_sources
    traits_all = a_traits + o_traits
    labels_all = a_labels + o_labels

    emb = embed_texts(texts_all)

    print("\n=== Per-trait ===")
    compare_per_trait(emb, sources_all, traits_all, labels_all, CONFIG["out_dir"])


if __name__ == "__main__":
    main()

