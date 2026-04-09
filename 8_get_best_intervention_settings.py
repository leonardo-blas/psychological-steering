import argparse
import ast
import math
import sqlite3
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from helpers import normalize_table_name


OCEAN = [
    ("O", "openness"),
    ("C", "conscientiousness"),
    ("E", "extraversion"),
    ("A", "agreeableness"),
    ("N", "neuroticism"),
]

CONFIG = {
    "results_path": "results",
    "cola_model_uncertainty": 0.05,
    "mean_fluency_delta_threshold": 0.05,
    "fluency_outlier_threshold_factor": 0.9,
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    return ap.parse_args()


def out_root_for(model: str, concept: str) -> Path:
    model_short = model.split("/")[-1]
    concept_norm = normalize_table_name(concept)
    return Path(CONFIG["results_path"]) / model_short / concept_norm


def mean(vals):
    total = 0.0
    count = 0
    for v in vals:
        total += float(v)
        count += 1
    if count == 0:
        return None
    return total / float(count)


def std(vals):
    mu = mean(vals)
    if mu is None:
        return None
    total = 0.0
    count = 0
    for v in vals:
        delta = float(v) - float(mu)
        total += delta * delta
        count += 1
    if count == 0:
        return None
    return math.sqrt(total / float(count))


def prob_to_1_5(p: float) -> float:
    p = float(p)
    if p < 0.0:
        return 1.0
    if p > 1.0:
        return 5.0
    return 1.0 + 4.0 * p


def q(x: Decimal, quantum: Decimal) -> Decimal:
    return x.quantize(quantum, rounding=ROUND_HALF_EVEN)


def fmt_beta(b: float) -> str:
    s = str(q(Decimal(str(b)), Decimal("0.000001"))).rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def _parse_first_beta(betas_str):
    try:
        xs = ast.literal_eval(betas_str)
    except Exception:
        return None
    if not xs:
        return None
    try:
        return float(xs[0])
    except Exception:
        return None


def sql_fetchone(db_path: Path, query: str, params=()):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(query, params)
        return cur.fetchone()


def sql_fetchall(db_path: Path, query: str, params=()):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(query, params)
        return cur.fetchall()


def require_table(db_path: Path, table: str):
    if not db_path.is_file():
        raise FileNotFoundError(f"Missing DB: {db_path}")
    row = sql_fetchone(
        db_path,
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;",
        (table,),
    )
    if row is None:
        raise RuntimeError(f'Expected table "{table}" not found in {db_path}')


def numeric_tables(db_path: Path):
    rows = sql_fetchall(db_path, "SELECT name FROM sqlite_master WHERE type='table';")
    out = []
    for (name,) in rows:
        if name.isdigit():
            out.append(int(name))
    out = sorted(set(out))
    if not out:
        raise RuntimeError(f"No numeric layer tables found in {db_path}")
    return out


def infer_num_layers_from_dbs(out_root: Path) -> int:
    sjt_db = out_root / "sjts_responses.db"
    inv_db = out_root / "inventory_responses.db"
    sjt_nums = numeric_tables(sjt_db)
    inv_nums = numeric_tables(inv_db)
    n = min(len(sjt_nums), len(inv_nums))
    for layer in range(n):
        require_table(sjt_db, str(layer))
        require_table(inv_db, str(layer))
    return int(n)


def load_layer_maps(out_root: Path, table_name: str):
    sjt_db = out_root / "sjts_responses.db"
    inv_db = out_root / "inventory_responses.db"

    sjt_rows = []
    inv_rows = []
    if sjt_db.is_file():
        require_table(sjt_db, table_name)
        sjt_rows = sql_fetchall(
            sjt_db,
            f'SELECT class, betas, concept_score, fluency_score FROM "{table_name}"',
        )
    if inv_db.is_file():
        require_table(inv_db, table_name)
        inv_rows = sql_fetchall(
            inv_db,
            f'SELECT class, betas, score FROM "{table_name}"',
        )

    sjt_base = []
    inv_base = []
    base_flu = []
    sjt_by = {"0": {}, "1": {}}
    inv_by = {"0": {}, "1": {}}
    flu_by = {"0": {}, "1": {}}

    for cls, betas_str, concept_score, fluency_score in sjt_rows:
        if cls is None:
            sjt_base.append(prob_to_1_5(float(concept_score)))
            base_flu.append(float(fluency_score))
            continue
        cls_s = str(int(cls))
        beta0 = _parse_first_beta(betas_str)
        if beta0 is None:
            continue
        beta = float(beta0)
        sjt_by[cls_s].setdefault(beta, []).append(prob_to_1_5(float(concept_score)))
        flu_by[cls_s].setdefault(beta, []).append(float(fluency_score))

    for cls, betas_str, score in inv_rows:
        if cls is None:
            inv_base.append(float(score))
            continue
        cls_s = str(int(cls))
        beta0 = _parse_first_beta(betas_str)
        if beta0 is None:
            continue
        beta = float(beta0)
        inv_by[cls_s].setdefault(beta, []).append(float(score))

    baseline_sjt = mean(sjt_base)
    baseline_inv = mean(inv_base)
    if not sjt_rows:
        baseline_sjt = None
    if not inv_rows:
        baseline_inv = None

    baseline_avg_flu = mean(v for v in base_flu if math.isfinite(float(v))) or 1.0
    cutoff = float(CONFIG["fluency_outlier_threshold_factor"]) * float(baseline_avg_flu)

    valid_betas = {"0": None, "1": None}
    for cls_s in ("0", "1"):
        if not flu_by[cls_s]:
            continue
        valid_betas[cls_s] = set()
        for beta in sorted(float(b) for b in flu_by[cls_s].keys()):
            flu_vals = [float(v) for v in flu_by[cls_s].get(beta, []) if math.isfinite(float(v))]
            if not flu_vals:
                break
            mean_flu = mean(flu_vals)
            if mean_flu is None:
                break

            outliers = 0
            for value in flu_vals:
                if float(value) < float(cutoff):
                    outliers += 1

            allowed = int(
                math.ceil(float(CONFIG["cola_model_uncertainty"]) * float(len(flu_vals)))
            )
            stop_outliers = outliers > allowed
            stop_mean = float(mean_flu) < (
                float(baseline_avg_flu)
                * (1.0 - float(CONFIG["mean_fluency_delta_threshold"]))
            )
            if stop_mean or stop_outliers:
                break
            valid_betas[cls_s].add(float(beta))

    sjt_mu = {"0": {}, "1": {}}
    inv_mu = {"0": {}, "1": {}}
    sjt_sigma = {"0": {}, "1": {}}
    inv_sigma = {"0": {}, "1": {}}

    for cls_s in ("0", "1"):
        allowed = valid_betas[cls_s]
        for beta, vals in sjt_by[cls_s].items():
            if allowed is not None and float(beta) not in allowed:
                continue
            mu = mean(vals)
            sigma = std(vals)
            if mu is not None:
                sjt_mu[cls_s][float(beta)] = float(mu)
            if sigma is not None:
                sjt_sigma[cls_s][float(beta)] = float(sigma)

        for beta, vals in inv_by[cls_s].items():
            if allowed is not None and float(beta) not in allowed:
                continue
            mu = mean(vals)
            sigma = std(vals)
            if mu is not None:
                inv_mu[cls_s][float(beta)] = float(mu)
            if sigma is not None:
                inv_sigma[cls_s][float(beta)] = float(sigma)

    return {
        "baseline_inv": None if baseline_inv is None else float(baseline_inv),
        "baseline_sjt": None if baseline_sjt is None else float(baseline_sjt),
        "inv_mu": inv_mu,
        "sjt_mu": sjt_mu,
        "inv_sigma": inv_sigma,
        "sjt_sigma": sjt_sigma,
    }


def pick_global_best(out_root: Path, num_layers: int, metric: str, baseline_mu: float):
    best_plus = None
    best_minus = None

    for layer in range(num_layers):
        data = load_layer_maps(out_root, str(layer))
        mu_map = data["inv_mu"] if metric == "inv" else data["sjt_mu"]
        sigma_map = data["inv_sigma"] if metric == "inv" else data["sjt_sigma"]

        for beta, mu in mu_map["1"].items():
            if float(mu) <= float(baseline_mu):
                continue
            candidate = (
                int(layer),
                float(beta),
                float(mu),
                float(sigma_map["1"].get(float(beta), 0.0)),
            )
            if (
                best_plus is None
                or candidate[2] > best_plus[2]
                or (candidate[2] == best_plus[2] and candidate[1] < best_plus[1])
                or (candidate[2] == best_plus[2] and candidate[1] == best_plus[1] and candidate[0] < best_plus[0])
            ):
                best_plus = candidate

        for beta, mu in mu_map["0"].items():
            if float(mu) >= float(baseline_mu):
                continue
            candidate = (
                int(layer),
                float(beta),
                float(mu),
                float(sigma_map["0"].get(float(beta), 0.0)),
            )
            if (
                best_minus is None
                or candidate[2] < best_minus[2]
                or (candidate[2] == best_minus[2] and candidate[1] < best_minus[1])
                or (candidate[2] == best_minus[2] and candidate[1] == best_minus[1] and candidate[0] < best_minus[0])
            ):
                best_minus = candidate

    return best_plus, best_minus


def best_line(best, baseline_mu: float) -> str:
    if best is None:
        return f"baseline (mu={baseline_mu:.3f})"
    layer, beta, mu, sigma = best
    return (
        f"layer={int(layer)} beta={fmt_beta(float(beta))} "
        f"mu={float(mu):.3f} sigma={float(sigma):.3f}"
    )


def main():
    args = parse_args()

    for _letter, concept in OCEAN:
        out_root = out_root_for(args.model, concept)
        if not out_root.is_dir():
            print(f"{concept}: missing results at {out_root}")
            continue

        sjt_db = out_root / "sjts_responses.db"
        inv_db = out_root / "inventory_responses.db"
        has_sjt = sjt_db.is_file()
        has_inv = inv_db.is_file()
        if not has_sjt and not has_inv:
            continue

        if has_sjt and has_inv:
            num_layers = infer_num_layers_from_dbs(out_root)
        elif has_sjt:
            num_layers = len(numeric_tables(sjt_db))
            for layer in range(num_layers):
                require_table(sjt_db, str(layer))
        else:
            num_layers = len(numeric_tables(inv_db))
            for layer in range(num_layers):
                require_table(inv_db, str(layer))

        base0 = load_layer_maps(out_root, "0")
        printed_header = False

        if has_inv:
            baseline_inv = float(base0["baseline_inv"])
            inv_plus, inv_minus = pick_global_best(out_root, num_layers, "inv", baseline_inv)
            if not printed_header:
                print(concept)
                printed_header = True
            print(f"  mpi120  up:   {best_line(inv_plus, baseline_inv)}")
            print(f"  mpi120  down: {best_line(inv_minus, baseline_inv)}")

        if has_sjt:
            baseline_sjt = float(base0["baseline_sjt"])
            sjt_plus, sjt_minus = pick_global_best(out_root, num_layers, "sjt", baseline_sjt)
            if not printed_header:
                print(concept)
                printed_header = True
            print(f"  sjts    up:   {best_line(sjt_plus, baseline_sjt)}")
            print(f"  sjts    down: {best_line(sjt_minus, baseline_sjt)}")


if __name__ == "__main__":
    main()
