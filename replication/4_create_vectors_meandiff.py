import argparse
import json
import os
import sqlite3
import numpy as np
import torch
from tqdm.auto import tqdm
from helpers import normalize_table_name


def get_mode_dir(mode: str) -> str:
    if mode == "b":
        return "binary_choice"
    if mode == "s":
        return "statement"
    raise ValueError(f"Unknown mode: {mode}")


def get_activations_db_path(model_name: str, mode: str) -> str:
    model_short = model_name.split("/")[-1]
    return f"data/{get_mode_dir(mode)}_activations/{model_short}.db"


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
        elif arr.shape != (L_ref, D_ref):
            raise ValueError(f"Shape mismatch: {arr.shape} vs {(L_ref, D_ref)}")
        X_list.append(arr)
        y_list.append(int(lbl))

    return np.stack(X_list, axis=0), np.array(y_list, dtype=int)


def run_meandiff(X: np.ndarray, y: np.ndarray):
    N, L, D = X.shape

    pos = y == 1
    neg = y == 0
    if not np.any(pos) or not np.any(neg):
        raise ValueError("Need both classes (0 and 1).")

    os.makedirs(OUT_DIR, exist_ok=True)
    distances = {}

    for layer in tqdm(range(L), desc=f"{CONCEPT}, meandiff"):
        X_l = X[:, layer, :]

        mu1 = X_l[pos].mean(axis=0)
        mu0 = X_l[neg].mean(axis=0)

        v = (mu1 - mu0).astype(np.float32)
        vn = float(np.linalg.norm(v))
        if vn <= 0.0:
            raise ValueError(f"Zero MeanDiff norm at layer {layer}.")

        u = (v / vn).astype(np.float32)
        x0 = 0.5 * (mu0 + mu1)

        torch.save(torch.from_numpy(v), os.path.join(OUT_DIR, f"layer_{layer}_raw.pt"))
        torch.save(torch.from_numpy(u), os.path.join(OUT_DIR, f"layer_{layer}.pt"))

        alphas = (X_l - x0.reshape(1, D)) @ u

        layer_out = {}
        for cls in (0, 1):
            m = y == cls
            if not np.any(m):
                continue

            cls_alphas = alphas[m]
            closest = float(cls_alphas[np.argmin(np.abs(cls_alphas))])
            furthest = float(cls_alphas[np.argmax(np.abs(cls_alphas))])
            centroid = float(cls_alphas.mean())

            layer_out[str(cls)] = {
                "closest": closest,
                "centroid": centroid,
                "furthest": furthest,
            }

        distances[str(layer)] = layer_out

    with open(os.path.join(OUT_DIR, "distances.json"), "w") as f:
        json.dump(distances, f, indent=2)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("-c", "--concept", required=True)
    ap.add_argument("--mode", required=True, choices=["b", "s"])
    return ap.parse_args()


def main():
    args = parse_args()

    global MODEL_DIR, CONCEPT, OUT_DIR
    MODEL_DIR = args.model.split("/")[-1]
    CONCEPT = args.concept

    table = normalize_table_name(CONCEPT)
    OUT_DIR = os.path.join("vectors", MODEL_DIR, table, "meandiff", get_mode_dir(args.mode))

    if os.path.exists(os.path.join(OUT_DIR, "distances.json")):
        return

    X, y = fetch_activations(get_activations_db_path(args.model, args.mode), table)
    run_meandiff(X, y)


if __name__ == "__main__":
    main()
