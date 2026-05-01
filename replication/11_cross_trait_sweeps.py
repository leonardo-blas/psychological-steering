import argparse
import json
import math
import sqlite3
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from helpers import seed_all, init_embed_model, embed_texts, init_fluency_model, fluency_scores
from psychometric_utils import run_sjts, run_inventory
from sweeping_utils import CONFIG, load_classifier


OCEAN = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


def mean(xs):
    s = 0.0
    for x in xs:
        s += float(x)
    return s / float(len(xs))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("-bs", "--batch_size", type=int, default=128)
    ap.add_argument("--quantize", "-q", action="store_true")
    return ap.parse_args()


def alphas_grid(alpha: float, x: float):
    out = []
    for i in range(10):
        a = (float(alpha) * float(i) / 9.0) * float(x)
        out.append([[float(a)]])
    return out


def pick_extrema(by_stride: dict, section: str, concept: str):
    hi = None
    lo = None
    for s in by_stride:
        d = by_stride[s][section][concept]
        rmax = d["max"]
        rmin = d["min"]

        cand_hi = (float(rmax["mu"]), int(rmax["layer"]), float(rmax["alpha"]), int(s))
        if hi is None or cand_hi[0] > hi[0]:
            hi = cand_hi

        cand_lo = (float(rmin["mu"]), int(rmin["layer"]), float(rmin["alpha"]), int(s))
        if lo is None or cand_lo[0] < lo[0]:
            lo = cand_lo

    return hi, lo


def init_dbs(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    sjt_db = sqlite3.connect(str(out_dir / "sjts_responses.db"))
    inv_db = sqlite3.connect(str(out_dir / "inventory_responses.db"))

    sjt_cur = sjt_db.cursor()
    inv_cur = inv_db.cursor()

    for t in OCEAN:
        sjt_cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {t} (
                class INTEGER,
                alpha REAL,
                sjt TEXT,
                answer TEXT,
                o REAL,
                c REAL,
                e REAL,
                a REAL,
                n REAL,
                fluency REAL
            )
            """
        )

        sjt_cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {t}_summary (
                class BOOLEAN,
                alpha REAL,
                fluency_invalid_value REAL,
                fluency_invalid_outliers INTEGER
            )
            """
        )

        inv_cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {t}_summary (
                class BOOLEAN,
                alpha REAL,
                fluency_invalid_value REAL,
                fluency_invalid_outliers INTEGER
            )
            """
        )

        inv_cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {t} (
                class INTEGER,
                alpha REAL,
                item TEXT,
                o REAL,
                c REAL,
                e REAL,
                a REAL,
                n REAL
            )
            """
        )

    sjt_db.commit()
    inv_db.commit()
    return sjt_db, inv_db


def main():
    args = parse_args()
    seed_all(int(CONFIG["seed"]))

    inventory = "mpi120"
    method = "meandiff"
    mode = "s"
    batch_size = int(args.batch_size)
    fit_intercept = False

    model_dir = Path(args.model).name
    analysis_dir = Path("analysis") / "sweeps" / model_dir
    extrema = json.loads((analysis_dir / "concept_extrema_mds.json").read_text(encoding="utf-8"))
    by_stride = extrema["by_stride"]

    out_dir = Path("results") / "entanglement" / model_dir
    sjt_db, inv_db = init_dbs(out_dir)
    sjt_cur = sjt_db.cursor()
    inv_cur = inv_db.cursor()

    embed_tok, embed_model = init_embed_model()

    fluency_tok, fluency_model = init_fluency_model()

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model_lm = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_cfg if args.quantize else None,
        low_cpu_mem_usage=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    probe_clfs = {}
    for c in OCEAN:
        probe_clfs[c] = load_classifier(c)

    col_i = {
        "openness": 0,
        "conscientiousness": 1,
        "extraversion": 2,
        "agreeableness": 3,
        "neuroticism": 4,
    }

    for inject_concept in OCEAN:
        hi_sjt, lo_sjt = pick_extrema(by_stride, "sjts", inject_concept)
        hi_inv, lo_inv = pick_extrema(by_stride, "mpi120", inject_concept)

        dist_path = Path("vectors") / model_dir / inject_concept / "meandiff" / "statement" / "distances.json"
        distances = json.loads(dist_path.read_text(encoding="utf-8"))

        for cls, pick in [(1, hi_sjt), (0, lo_sjt)]:
            layer = int(pick[1])
            alpha_max = float(pick[2])
            stride = int(pick[3])
            if layer == 0 and alpha_max == 0.0:
                continue
            layers = [layer]

            x = float(distances[str(layer)]["1"]["centroid"])

            grid = alphas_grid(alpha_max, x)
            if cls == 0:
                grid = [[[-v[0][0]]] for v in grid]

            _, ans0 = run_sjts(
                model=model_lm,
                tokenizer=tokenizer,
                inventory=inventory,
                method=method,
                concepts=[inject_concept],
                layers=layers,
                model_name=args.model,
                fit_intercept=fit_intercept,
                alphas=[[0.0]],
                mode=mode,
                batch_size=batch_size,
                stride=stride,
            )

            texts0 = []
            for t in ans0:
                texts0.append(str(t or "").strip())

            base_flu_probs = fluency_scores(
                fluency_tok,
                fluency_model,
                texts0,
                batch_size=int(CONFIG["fluency_batch_size"]),
            )
            baseline_avg_flu = mean(base_flu_probs)
            cutoff = float(CONFIG["fluency_outlier_threshold_factor"]) * float(baseline_avg_flu)

            for alpha_setting in grid:
                alpha = float(alpha_setting[0][0])
                alpha = 0.0 if x == 0.0 else (alpha / x)

                sjt_prompts, ans = run_sjts(
                    model=model_lm,
                    tokenizer=tokenizer,
                    inventory=inventory,
                    method=method,
                    concepts=[inject_concept],
                    layers=layers,
                    model_name=args.model,
                    fit_intercept=fit_intercept,
                    alphas=alpha_setting,
                    mode=mode,
                    batch_size=batch_size,
                    stride=stride,
                )

                texts = []
                for t in ans:
                    texts.append(str(t or "").strip())

                flu_probs = fluency_scores(
                    fluency_tok,
                    fluency_model,
                    texts,
                    batch_size=int(CONFIG["fluency_batch_size"]),
                )
                mean_flu = mean(flu_probs)

                outliers = 0
                for v in flu_probs:
                    if float(v) < float(cutoff):
                        outliers += 1

                allowed = int(math.ceil(float(CONFIG["fluency_model_uncertainty"]) * float(len(flu_probs))))
                stop_outliers = outliers > allowed
                stop_mean = float(mean_flu) < (
                    float(baseline_avg_flu) * (1.0 - float(CONFIG["mean_fluency_delta_threshold"]))
                )

                fluency_invalid_value = float(mean_flu) if stop_mean else 0.0
                fluency_invalid_outliers = int(outliers - allowed) if stop_outliers else 0

                sjt_cur.execute(
                    f"""
                    INSERT INTO {inject_concept}_summary
                    (class, alpha, fluency_invalid_value, fluency_invalid_outliers)
                    VALUES (?, ?, ?, ?)
                    """,
                    (bool(cls), float(alpha), float(fluency_invalid_value), int(fluency_invalid_outliers)),
                )

                X = embed_texts(
                    embed_tok,
                    embed_model,
                    texts,
                    batch_size=int(CONFIG["probe_embed_batch"]),
                )

                per_trait_scores = {}
                for c in OCEAN:
                    clf = probe_clfs[c]
                    probs = clf.predict_proba(X)
                    pos_index = list(clf.classes_).index(1)
                    per_trait_scores[c] = [float(v) for v in probs[:, pos_index]]

                sjt_rows = []
                for i in range(len(texts)):
                    sjt_rows.append(
                        (
                            int(cls),
                            float(alpha),
                            str(sjt_prompts[i]),
                            str(texts[i]),
                            float(per_trait_scores["openness"][i]),
                            float(per_trait_scores["conscientiousness"][i]),
                            float(per_trait_scores["extraversion"][i]),
                            float(per_trait_scores["agreeableness"][i]),
                            float(per_trait_scores["neuroticism"][i]),
                            float(flu_probs[i]),
                        )
                    )

                sjt_cur.executemany(
                    f"""
                    INSERT INTO {inject_concept}
                    (class, alpha, sjt, answer, o, c, e, a, n, fluency)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    sjt_rows,
                )
                sjt_db.commit()

        for cls, pick in [(1, hi_inv), (0, lo_inv)]:
            layer = int(pick[1])
            alpha_max = float(pick[2])
            stride = int(pick[3])
            if layer == 0 and alpha_max == 0.0:
                continue
            layers = [layer]

            x = float(distances[str(layer)]["1"]["centroid"])

            grid = alphas_grid(alpha_max, x)
            if cls == 0:
                grid = [[[-v[0][0]]] for v in grid]

            _, ans0 = run_sjts(
                model=model_lm,
                tokenizer=tokenizer,
                inventory=inventory,
                method=method,
                concepts=[inject_concept],
                layers=layers,
                model_name=args.model,
                fit_intercept=fit_intercept,
                alphas=[[0.0]],
                mode=mode,
                batch_size=batch_size,
                stride=stride,
            )

            texts0 = []
            for t in ans0:
                texts0.append(str(t or "").strip())

            base_flu_probs = fluency_scores(
                fluency_tok,
                fluency_model,
                texts0,
                batch_size=int(CONFIG["fluency_batch_size"]),
            )
            baseline_avg_flu = mean(base_flu_probs)
            cutoff = float(CONFIG["fluency_outlier_threshold_factor"]) * float(baseline_avg_flu)

            for alpha_setting in grid:
                alpha = float(alpha_setting[0][0])
                alpha = 0.0 if x == 0.0 else (alpha / x)

                _, ans = run_sjts(
                    model=model_lm,
                    tokenizer=tokenizer,
                    inventory=inventory,
                    method=method,
                    concepts=[inject_concept],
                    layers=layers,
                    model_name=args.model,
                    fit_intercept=fit_intercept,
                    alphas=alpha_setting,
                    mode=mode,
                    batch_size=batch_size,
                    stride=stride,
                )

                texts = []
                for t in ans:
                    texts.append(str(t or "").strip())

                flu_probs = fluency_scores(
                    fluency_tok,
                    fluency_model,
                    texts,
                    batch_size=int(CONFIG["fluency_batch_size"]),
                )
                mean_flu = mean(flu_probs)

                outliers = 0
                for v in flu_probs:
                    if float(v) < float(cutoff):
                        outliers += 1

                allowed = int(math.ceil(float(CONFIG["fluency_model_uncertainty"]) * float(len(flu_probs))))
                stop_outliers = outliers > allowed
                stop_mean = float(mean_flu) < (
                    float(baseline_avg_flu) * (1.0 - float(CONFIG["mean_fluency_delta_threshold"]))
                )

                fluency_invalid_value = float(mean_flu) if stop_mean else 0.0
                fluency_invalid_outliers = int(outliers - allowed) if stop_outliers else 0

                inv_cur.execute(
                    f"""
                    INSERT INTO {inject_concept}_summary
                    (class, alpha, fluency_invalid_value, fluency_invalid_outliers)
                    VALUES (?, ?, ?, ?)
                    """,
                    (bool(cls), float(alpha), float(fluency_invalid_value), int(fluency_invalid_outliers)),
                )
                inv_db.commit()

                alphas_eval = []
                for c in OCEAN:
                    if c == inject_concept:
                        alphas_eval.append(alpha_setting[0])
                    else:
                        alphas_eval.append([0.0])

                dims, items_scored, scores = run_inventory(
                    model=model_lm,
                    tokenizer=tokenizer,
                    inventory=inventory,
                    method=method,
                    concepts=OCEAN,
                    evaluation_concepts=OCEAN,
                    layers=layers,
                    model_name=args.model,
                    fit_intercept=fit_intercept,
                    alphas=alphas_eval,
                    mode=mode,
                    batch_size=batch_size,
                    stride=stride,
                )

                if isinstance(dims, str):
                    dims = [dims] * len(items_scored)
                if not isinstance(scores, (list, tuple)):
                    scores = [scores] * len(items_scored)

                inv_rows = []
                for i in range(len(items_scored)):
                    vals = [None, None, None, None, None]
                    vals[col_i[dims[i]]] = float(scores[i])
                    inv_rows.append(
                        (
                            int(cls),
                            float(alpha),
                            str(items_scored[i]),
                            vals[0],
                            vals[1],
                            vals[2],
                            vals[3],
                            vals[4],
                        )
                    )

                inv_cur.executemany(
                    f"""
                    INSERT INTO {inject_concept}
                    (class, alpha, item, o, c, e, a, n)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    inv_rows,
                )
                inv_db.commit()


if __name__ == "__main__":
    main()
