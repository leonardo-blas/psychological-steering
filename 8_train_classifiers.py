import sqlite3
from pathlib import Path
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
from helpers import seed_all, embed_batch, init_embed_model


CONFIG = {
    "seed": 42,
    "probe_db_path": "data/statements.db",
    "max_iter": 1000,
    "tol": 0.001,
}


def classifier_path(name: str) -> Path:
    root = Path("classifiers")
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{name}.pkl"


def get_tables():
    with sqlite3.connect(CONFIG["probe_db_path"]) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        rows = cur.fetchall()
    return [r[0] for r in rows]


def train_classifier_for_table(table: str, embed_tok, embed_model):
    texts = []
    labels = []
    with sqlite3.connect(CONFIG["probe_db_path"]) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT statement, label FROM {table}")
        rows = cur.fetchall()
        for s, lbl in rows:
            if not s:
                continue
            texts.append(s.strip())
            labels.append(int(lbl))
    if not texts:
        raise ValueError(f"Cannot train classifier for {table}: no data.")
    if len(set(labels)) < 2:
        raise ValueError(f"Cannot train classifier for {table}: only one class.")

    X = embed_batch(embed_tok, embed_model, texts).to(torch.float32).cpu().numpy()
    y = torch.tensor(labels).numpy()

    clf = LogisticRegression(
        max_iter=CONFIG["max_iter"],
        tol=CONFIG["tol"],
    )
    clf.fit(X, y)

    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y,
        y_pred,
        average="binary",
        zero_division=0,
    )
    print(
        f"Classifier metrics ({table}): "
        f"acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}"
    )

    path = classifier_path(table)
    joblib.dump(clf, path)
    print(f"Saved classifier to {path}")


def main():
    seed_all(CONFIG["seed"])
    tables = get_tables()
    print(f"Found tables: {', '.join(tables)}")
    embed_tok, embed_model = init_embed_model()
    for table in tables:
        print(f"Training classifier for table {table}...")
        train_classifier_for_table(table, embed_tok, embed_model)


if __name__ == "__main__":
    main()
