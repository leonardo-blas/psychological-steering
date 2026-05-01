import argparse
import ast
import math
import sqlite3
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

from helpers import normalize_table_name


CONFIG = {
    "results_path": "results/stride_1/step_1/",
    "cola_model_uncertainty": 0.05,
    "mean_fluency_delta_threshold": 0.05,
    "fluency_outlier_threshold_factor": 0.9,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print the best valid SJT layer and coefficient per trait and direction."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model id, e.g. meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "-c",
        "--concept",
        required=True,
        help='Concept to report, e.g. "openness".',
    )
    return parser.parse_args()


def out_root_for(model: str, concept: str) -> Path:
    model_short = model.split("/")[-1]
    concept_norm = normalize_table_name(concept)
    return Path(CONFIG["results_path"]) / model_short / concept_norm / "meandiff/statement/"


def mean(vals):
    total = 0.0
    count = 0
    for value in vals:
        total += float(value)
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
    for value in vals:
        delta = float(value) - float(mu)
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


def fmt_alpha(alpha: float) -> str:
    s = str(q(Decimal(str(alpha)), Decimal("0.000001"))).rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def _parse_first_alpha(betas_str):
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


def load_layer_maps(out_root: Path, table_name: str):
    sjt_db = out_root / "sjts_responses.db"

    require_table(sjt_db, table_name)
    sjt_rows = sql_fetchall(
        sjt_db,
        f'SELECT class, betas, concept_score, fluency_score FROM "{table_name}"',
    )

    sjt_base = []
    base_flu = []
    sjt_by = {"0": {}, "1": {}}
    flu_by = {"0": {}, "1": {}}

    for cls, betas_str, concept_score, fluency_score in sjt_rows:
        if cls is None:
            sjt_base.append(prob_to_1_5(float(concept_score)))
            base_flu.append(float(fluency_score))
            continue

        cls_s = str(int(cls))
        alpha0 = _parse_first_alpha(betas_str)
        if alpha0 is None:
            continue

        alpha = float(alpha0)
        sjt_by[cls_s].setdefault(alpha, []).append(prob_to_1_5(float(concept_score)))
        flu_by[cls_s].setdefault(alpha, []).append(float(fluency_score))

    baseline_sjt = mean(sjt_base)
    baseline_avg_flu = mean(v for v in base_flu if math.isfinite(float(v))) or 1.0
    cutoff = float(CONFIG["fluency_outlier_threshold_factor"]) * float(baseline_avg_flu)

    valid_betas = {"0": None, "1": None}
    for cls_s in ("0", "1"):
        if not flu_by[cls_s]:
            continue

        valid_betas[cls_s] = set()
        for alpha in sorted(float(b) for b in flu_by[cls_s].keys()):
            flu_vals = [float(v) for v in flu_by[cls_s].get(alpha, []) if math.isfinite(float(v))]
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
                float(baseline_avg_flu) * (1.0 - float(CONFIG["mean_fluency_delta_threshold"]))
            )

            if stop_mean or stop_outliers:
                break

            valid_betas[cls_s].add(float(alpha))

    sjt_mu = {"0": {}, "1": {}}
    sjt_sigma = {"0": {}, "1": {}}

    for cls_s in ("0", "1"):
        allowed = valid_betas[cls_s]
        for alpha, vals in sjt_by[cls_s].items():
            if allowed is not None and float(alpha) not in allowed:
                continue

            mu = mean(vals)
            sigma = std(vals)

            if mu is not None:
                sjt_mu[cls_s][float(alpha)] = float(mu)
            if sigma is not None:
                sjt_sigma[cls_s][float(alpha)] = float(sigma)

    return {
        "baseline_sjt": None if baseline_sjt is None else float(baseline_sjt),
        "sjt_mu": sjt_mu,
        "sjt_sigma": sjt_sigma,
    }


def pick_global_best(out_root: Path, num_layers: int, baseline_mu: float):
    best_plus = None
    best_minus = None

    for layer in range(num_layers):
        data = load_layer_maps(out_root, str(layer))
        mu_map = data["sjt_mu"]
        sigma_map = data["sjt_sigma"]

        for alpha, mu in mu_map["1"].items():
            if float(mu) <= float(baseline_mu):
                continue

            candidate = (
                int(layer),
                float(alpha),
                float(mu),
                float(sigma_map["1"].get(float(alpha), 0.0)),
            )

            if (
                best_plus is None
                or candidate[2] > best_plus[2]
                or (candidate[2] == best_plus[2] and candidate[1] < best_plus[1])
                or (
                    candidate[2] == best_plus[2]
                    and candidate[1] == best_plus[1]
                    and candidate[0] < best_plus[0]
                )
            ):
                best_plus = candidate

        for alpha, mu in mu_map["0"].items():
            if float(mu) >= float(baseline_mu):
                continue

            candidate = (
                int(layer),
                float(alpha),
                float(mu),
                float(sigma_map["0"].get(float(alpha), 0.0)),
            )

            if (
                best_minus is None
                or candidate[2] < best_minus[2]
                or (candidate[2] == best_minus[2] and candidate[1] < best_minus[1])
                or (
                    candidate[2] == best_minus[2]
                    and candidate[1] == best_minus[1]
                    and candidate[0] < best_minus[0]
                )
            ):
                best_minus = candidate

    return best_plus, best_minus


def format_best(best, coefficient_sign: int = 1):
    if best is None:
        return "no valid setting found"
    layer, alpha, mu, sigma = best
    alpha = abs(float(alpha)) * float(coefficient_sign)
    return (
        f"layer={int(layer)} coefficient={fmt_alpha(alpha)} "
        f"mu={float(mu):.3f} sigma={float(sigma):.3f}"
    )


def main():
    args = parse_args()

    concept = args.concept
    out_root = out_root_for(args.model, concept)
    sjt_db = out_root / "sjts_responses.db"

    num_layers = len(numeric_tables(sjt_db))
    for layer in range(num_layers):
        require_table(sjt_db, str(layer))

    base0 = load_layer_maps(out_root, "0")
    baseline_sjt = float(base0["baseline_sjt"])
    best_up, best_down = pick_global_best(out_root, num_layers, baseline_sjt)

    print(concept)
    print(f"  up:   {format_best(best_up)}")
    print(f"  down: {format_best(best_down, coefficient_sign=-1)}")


if __name__ == "__main__":
    main()
