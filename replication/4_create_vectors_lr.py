import argparse
import json
import os
import sqlite3
import warnings
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm.auto import tqdm
from helpers import normalize_table_name


CONFIG = {
    "seed": 42,
}
warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in log",
    category=RuntimeWarning,
    module="sklearn.linear_model._logistic",
)


def get_mode_dir(mode: str) -> str:
    if mode == "b":
        return "binary_choice"
    if mode == "s":
        return "statement"
    raise ValueError(f"Unknown mode: {mode}")


def get_activations_db_path(model_name: str, mode: str) -> str:
    model_short = model_name.split("/")[-1]
    mode_dir = get_mode_dir(mode)
    return f"data/{mode_dir}_activations/{model_short}.db"


def fetch_activations(db_path: str, table: str):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT activations, label FROM {table}")
        rows = cur.fetchall()

    X_list = []
    y_list = []
    L_ref = None
    D_ref = None

    for payload, lbl in rows:
        arr = np.array(json.loads(payload), dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected [L,D], got {arr.shape}")
        if L_ref is None:
            L_ref, D_ref = arr.shape
        else:
            if arr.shape != (L_ref, D_ref):
                raise ValueError(
                    f"Shape mismatch: {arr.shape} vs {(L_ref, D_ref)}"
                )
        X_list.append(arr)
        y_list.append(int(lbl))

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=int)
    return torch.from_numpy(X), torch.from_numpy(y)


def run_probe(X, y, reg_type, fit_intercept):
    X = X.numpy()
    y = y.numpy()
    L = X.shape[1]

    y_tr_all = y
    val_probs_agg = np.zeros_like(y_tr_all, dtype=np.float64)

    train_metrics = {}
    distances_by_layer = {}

    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    desc = "fitted intercept" if fit_intercept else "zero intercept"
    desc = f"{CONCEPT}, {reg_type.upper()} {desc}"
    for layer in tqdm(range(L), desc=desc):
        X_l = X[:, layer, :]

        X_tr = X_l
        y_tr = y_tr_all

        best_clf = None
        best_C = None
        best_acc_tr = -1.0
        best_clf_any = None
        best_C_any = None

        for C in Cs:
            clf = LogisticRegression(
                C=C,
                penalty=reg_type,
                solver="saga" if reg_type == "l1" else "lbfgs",
                fit_intercept=fit_intercept,
                tol=0.001,
                max_iter=10000,
                random_state=CONFIG["seed"],
            )
            clf.fit(X_tr, y_tr)

            acc_tr_c = clf.score(X_tr, y_tr)

            if acc_tr_c > best_acc_tr:
                best_acc_tr = acc_tr_c
                best_clf_any = clf
                best_C_any = C

            if acc_tr_c == 1.0:
                best_clf = clf
                best_C = C
                break

        if best_clf is None:
            best_clf = best_clf_any
            best_C = best_C_any

        y_tr_pred = best_clf.predict(X_tr)
        y_tr_prob = best_clf.predict_proba(X_tr)[:, 1]
        acc_tr_layer = best_clf.score(X_tr, y_tr)

        prec_tr_val, rec_tr_val, f1_tr_val, _ = precision_recall_fscore_support(
            y_tr, y_tr_pred, average="binary", zero_division=0
        )

        try:
            auc_tr_val = roc_auc_score(y_tr, y_tr_prob)
        except ValueError:
            auc_tr_val = float("nan")

        mse_tr_val = mean_squared_error(y_tr, y_tr_prob)

        layer_key = int(layer)

        train_metrics[layer_key] = {
            "acc": float(acc_tr_layer),
            "precision": float(prec_tr_val),
            "recall": float(rec_tr_val),
            "f1": float(f1_tr_val),
            "auc": float(auc_tr_val),
            "mse": float(mse_tr_val),
        }

        val_probs_agg = val_probs_agg + y_tr_prob

        w = torch.from_numpy(best_clf.coef_).squeeze(0)
        b = torch.tensor(best_clf.intercept_[0], dtype=w.dtype)
        wn = w.norm(p=2)
        u = w / wn
        b_u = b / wn

        c_pos = torch.from_numpy(X_l[y == 1]).to(dtype=u.dtype).mean(0)
        d = torch.dot(u, c_pos) + b_u
        if d.item() < 0:
            u = -u
            b_u = -b_u

        os.makedirs(OUT_DIR, exist_ok=True)
        torch.save(
            torch.cat([w, b.view(1)], 0),
            os.path.join(OUT_DIR, f"layer_{layer}_C_{best_C}_wb.pt"),
        )
        torch.save(u, os.path.join(OUT_DIR, f"layer_{layer}_C_{best_C}.pt"))

        u_np = u.cpu().numpy()
        b_u_np = float(b_u.cpu().item())
        d_all_tr = X_tr @ u_np + b_u_np

        dist_layer = {}
        for cls in (0, 1):
            mask = y_tr == cls
            if not np.any(mask):
                continue
            d_cls = d_all_tr[mask]
            closest = float(d_cls[np.argmin(np.abs(d_cls))])
            furthest = float(d_cls[np.argmax(np.abs(d_cls))])
            centroid = float(d_cls.mean())
            dist_layer[int(cls)] = {
                "closest": closest,
                "centroid": centroid,
                "furthest": furthest,
            }
        distances_by_layer[layer_key] = dist_layer

    val_probs_agg = val_probs_agg / float(L)
    val_preds_agg = (val_probs_agg >= 0.5).astype(int)
    acc_val_agg = float((val_preds_agg == y_tr_all).mean() * 100.0)

    prec_val_agg, rec_val_agg, f1_val_agg, _ = precision_recall_fscore_support(
        y_tr_all, val_preds_agg, average="binary", zero_division=0
    )

    try:
        auc_val_agg = roc_auc_score(y_tr_all, val_probs_agg)
    except ValueError:
        auc_val_agg = float("nan")

    mse_val_agg = mean_squared_error(y_tr_all, val_probs_agg)

    train_metrics["aggregation"] = {
        "acc": float(acc_val_agg),
        "precision": float(prec_val_agg),
        "recall": float(rec_val_agg),
        "f1": float(f1_val_agg),
        "auc": float(auc_val_agg),
        "mse": float(mse_val_agg),
    }

    with open(os.path.join(OUT_DIR, "train_metrics.json"), "w") as f:
        json.dump(train_metrics, f, indent=2)

    with open(os.path.join(OUT_DIR, "distances.json"), "w") as f:
        json.dump(distances_by_layer, f, indent=2)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("-c", "--concept", required=True)
    ap.add_argument("-r", "--regularization", choices=["l1", "l2"], required=True)
    ap.add_argument("-i", "--fit_intercept", action="store_true")
    ap.add_argument(
        "--mode",
        required=True,
        choices=["b", "s"],
    )
    return ap.parse_args()


def main():
    args = parse_args()

    global MODEL_DIR, CONCEPT, OUT_DIR
    MODEL_DIR = args.model.split("/")[-1]
    CONCEPT = args.concept
    table = normalize_table_name(CONCEPT)
    mode_dir = get_mode_dir(args.mode)
    method_dir = (
        f"{args.regularization}_"
        f"{'fitted_intercept' if args.fit_intercept else 'zero_intercept'}"
    )
    OUT_DIR = os.path.join(
        "vectors",
        MODEL_DIR,
        table,
        method_dir,
        mode_dir,
    )

    needed_files = [
        "train_metrics.json",
        "distances.json",
    ]
    all_exist = True
    for filename in needed_files:
        full_path = os.path.join(OUT_DIR, filename)
        if not os.path.exists(full_path):
            all_exist = False
            break
    if all_exist:
        return

    db_path = get_activations_db_path(args.model, args.mode)
    X, y = fetch_activations(db_path, table)
    run_probe(X, y, args.regularization, args.fit_intercept)


if __name__ == "__main__":
    main()
