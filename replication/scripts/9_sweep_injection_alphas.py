import argparse
import math
import sqlite3
from decimal import Decimal
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from helpers import (
    seed_all,
    init_embed_model,
    embed_texts,
    init_fluency_model,
    fluency_scores
)
from psychometric_utils import run_sjts, run_inventory
from sweeping_utils import (
    CONFIG,
    validate_args,
    parse_layers_arg,
    compute_layers_suffix,
    filter_layer_groups_perfect,
    load_distances,
    load_metrics,
    load_classifier,
    get_output_root,
    group_label,
    write_sqlite,
    write_inventory_sqlite,
)
from data.inventory_to_constructs import inventory_to_dimensions


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--method", choices=["l1", "l2", "l1_and_l2", "meandiff"], required=True)
    ap.add_argument("--mode", choices=["b", "s"], required=True)
    ap.add_argument("-q", "--quantize", action="store_true")
    ap.add_argument("-i", "--fit_intercept", action="store_true")
    ap.add_argument("-bs", "--batch_size", type=int, required=True)
    ap.add_argument("--stride", type=int, required=True)
    return ap.parse_args()


def prob_to_1_5(p: float) -> float:
    p = float(p)
    if p < 0.0:
        return 1.0
    if p > 1.0:
        return 5.0
    return 1.0 + 4.0 * p


def mean(vals):
    s = 0.0
    for v in vals:
        s += float(v)
    return s / float(len(vals))


def log_io(msg: str):
    print(f"[IO] {msg}", flush=True)


def job_done(args_job, layer_groups):
    out_root = get_output_root(args_job)
    sjt_db = out_root / "sjts_responses.db"
    inv_db = out_root / "inventory_responses.db"

    for layers_here in layer_groups:
        name = group_label(args_job, layers_here)
        if not sjt_db.is_file():
            return False
        if not inv_db.is_file():
            return False

        with sqlite3.connect(sjt_db) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
                (name,),
            )
            if cur.fetchone() is None:
                return False

        with sqlite3.connect(inv_db) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
                (name,),
            )
            if cur.fetchone() is None:
                return False

    return True


def main():
    args = parse_args()
    stride = args.stride
    if stride <= 0:
        raise ValueError("--stride must be > 0.")

    inventories_map = inventory_to_dimensions

    if args.inventory not in inventories_map:
        raise ValueError(
            f"Unknown inventory {args.inventory!r}. "
            f"Available: {', '.join(sorted(inventories_map.keys()))}"
        )

    methods = ["l1", "l2"] if args.method == "l1_and_l2" else [args.method]
    modes = [args.mode]

    embed_tok, embed_model = init_embed_model()

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
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

    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:
        num_layers = getattr(model.config, "num_layers", None)
    if num_layers is None:
        enc = tokenizer("hi", return_tensors="pt")
        for k in enc:
            enc[k] = enc[k].to(model.device)
        hs = model(**enc, output_hidden_states=True, return_dict=True).hidden_states
        num_layers = len(hs) - 1

    num_layers = int(num_layers)
    all_layers = [-1] + list(range(num_layers))

    fluency_tok, fluency_model = init_fluency_model()

    concepts = inventories_map[args.inventory]

    for concept in concepts:
        for method in methods:
            for mode in modes:
                seed_all(int(CONFIG["seed"]))

                distances = load_distances(
                    model_name=args.model,
                    concept=concept,
                    method=method,
                    fit_intercept=bool(args.fit_intercept),
                    mode=mode,
                )

                train_metrics = None
                test_metrics = None
                if method != "meandiff":
                    train_metrics, test_metrics = load_metrics(
                        model_name=args.model,
                        concept=concept,
                        method=method,
                        fit_intercept=bool(args.fit_intercept),
                        mode=mode,
                    )

                per_layer_cls_x = {}
                for L_str, layer_info in distances.items():
                    L_int = int(L_str)
                    per_layer_cls_x[L_int] = {}
                    for cls_s in ("0", "1"):
                        cls_info = layer_info.get(cls_s)
                        if cls_info is None:
                            raise KeyError(f'Class "{cls_s}" missing for layer {L_int} in distances.json')
                        centroid = cls_info.get("centroid", None)
                        if centroid is None:
                            raise KeyError(f'"centroid" missing for layer {L_int}, class {cls_s} in distances.json')
                        per_layer_cls_x[L_int][cls_s] = float(centroid)

                probe_clf = load_classifier(concept)

                # ---- MAIN SWEEP ----
                for layer in all_layers:
                    layers_arg = f"[{layer}]"

                    seed_all(int(CONFIG["seed"]))

                    args_job = argparse.Namespace()
                    args_job.model = args.model
                    args_job.concept = concept
                    args_job.method = method
                    args_job.fit_intercept = bool(args.fit_intercept)
                    args_job.mode = mode
                    args_job.inventory = args.inventory
                    args_job.layers = layers_arg
                    args_job.perfect_only = (method != "meandiff")
                    args_job.fluency_constrained = True
                    args_job.start = None
                    args_job.end = None
                    args_job.step = 1
                    args_job.stride = stride
                    args_job.quantize = bool(args.quantize)
                    args_job.batch_size = int(args.batch_size)

                    validate_args(args_job)

                    layer_groups = parse_layers_arg(args_job.layers, num_layers)
                    if method != "meandiff":
                        layer_groups = filter_layer_groups_perfect(layer_groups, distances, train_metrics, test_metrics)

                    if not layer_groups:
                        if method == "meandiff":
                            print("No layers requested after parsing. Skipping.")
                        else:
                            print("No requested layers have acc == 1.0 on both train and test. Skipping.")
                        continue

                    args_job.layers_suffix = compute_layers_suffix(layer_groups, num_layers)

                    if job_done(args_job, layer_groups):
                        print("All requested layer groups already processed; skipping.")
                        continue

                    results = []
                    inv_results = []

                    for group_index, resolved_layers in enumerate(layer_groups):
                        zero_alphas_for_group = [0.0 for _ in resolved_layers]

                        baseline_questions, sjt_answers = run_sjts(
                            model=model,
                            tokenizer=tokenizer,
                            inventory=args_job.inventory,
                            method=args_job.method,
                            concepts=[args_job.concept],
                            layers=resolved_layers,
                            model_name=args_job.model,
                            fit_intercept=args_job.fit_intercept,
                            alphas=[zero_alphas_for_group],
                            mode=args_job.mode,
                            batch_size=args_job.batch_size,
                            stride=args_job.stride,
                        )

                        baseline_entries = []
                        for q_text, a_text in zip(baseline_questions, sjt_answers):
                            entry = {
                                "group": group_index,
                                "cls": "baseline",
                                "sign": "0",
                                "question": q_text,
                                "text": a_text,
                                "alpha_factor": 0.0,
                                "fluency": 0.0,
                            }
                            baseline_entries.append(entry)
                            results.append(entry)

                        _, inv_items, inv_scores = run_inventory(
                            model=model,
                            tokenizer=tokenizer,
                            inventory=args_job.inventory,
                            method=args_job.method,
                            concepts=[args_job.concept],
                            layers=resolved_layers,
                            model_name=args_job.model,
                            fit_intercept=args_job.fit_intercept,
                            alphas=[zero_alphas_for_group],
                            mode=args_job.mode,
                            batch_size=args_job.batch_size,
                            stride=args_job.stride,
                        )

                        for item, score in zip(inv_items, inv_scores):
                            inv_results.append(
                                {
                                    "group": group_index,
                                    "cls": "baseline",
                                    "sign": "0",
                                    "item": item,
                                    "score": float(score),
                                    "alpha_factor": 0.0,
                                }
                            )

                        if baseline_entries:
                            baseline_texts = [e["text"].strip() for e in baseline_entries]
                            baseline_scores = fluency_scores(
                                fluency_tok,
                                fluency_model,
                                baseline_texts,
                                batch_size=int(CONFIG["fluency_batch_size"]),
                            )
                            for e, sc in zip(baseline_entries, baseline_scores):
                                e["fluency"] = float(sc)
                            baseline_avg_fluency = mean([float(e["fluency"]) for e in baseline_entries])
                        else:
                            baseline_avg_fluency = 1.0

                        for cls_s in ("0", "1"):
                            step_dec = Decimal(str(args_job.step))
                            step_index = 1
                            sjt_sig_hist = []
                            inv_sig_hist = []

                            while True:
                                alpha_factor = float(step_dec * Decimal(step_index))
                                alphas_for_group = [alpha_factor * per_layer_cls_x[L][cls_s] for L in resolved_layers]

                                sjt_questions, sjt_answers = run_sjts(
                                    model=model,
                                    tokenizer=tokenizer,
                                    inventory=args_job.inventory,
                                    method=args_job.method,
                                    concepts=[args_job.concept],
                                    layers=resolved_layers,
                                    model_name=args_job.model,
                                    fit_intercept=args_job.fit_intercept,
                                    alphas=[alphas_for_group],
                                    mode=args_job.mode,
                                    batch_size=args_job.batch_size,
                                    stride=args_job.stride,
                                )

                                if not sjt_answers:
                                    break

                                batch_entries = []
                                for q_text, a_text in zip(sjt_questions, sjt_answers):
                                    entry = {
                                        "group": group_index,
                                        "cls": cls_s,
                                        "sign": "+" if cls_s == "1" else "-",
                                        "question": q_text,
                                        "text": a_text,
                                        "alpha_factor": alpha_factor,
                                        "fluency": 0.0,
                                    }
                                    batch_entries.append(entry)
                                    results.append(entry)

                                _, inv_items, inv_scores = run_inventory(
                                    model=model,
                                    tokenizer=tokenizer,
                                    inventory=args_job.inventory,
                                    method=args_job.method,
                                    concepts=[args_job.concept],
                                    layers=resolved_layers,
                                    model_name=args_job.model,
                                    fit_intercept=args_job.fit_intercept,
                                    alphas=[alphas_for_group],
                                    mode=args_job.mode,
                                    batch_size=args_job.batch_size,
                                    stride=args_job.stride,
                                )

                                for item, score in zip(inv_items, inv_scores):
                                    inv_results.append(
                                        {
                                            "group": group_index,
                                            "cls": cls_s,
                                            "sign": "+" if cls_s == "1" else "-",
                                            "item": item,
                                            "score": float(score),
                                            "alpha_factor": alpha_factor,
                                        }
                                    )

                                texts_for_fluency = [e["text"].strip() for e in batch_entries]
                                scores = fluency_scores(
                                    fluency_tok,
                                    fluency_model,
                                    texts_for_fluency,
                                    batch_size=int(CONFIG["fluency_batch_size"]),
                                )
                                for e, sc in zip(batch_entries, scores):
                                    e["fluency"] = float(sc)

                                sjt_sig = tuple(
                                    (str(q or "").strip(), str(a or "").strip())
                                    for q, a in zip(sjt_questions, sjt_answers)
                                )
                                inv_sig = tuple(
                                    (str(it or "").strip(), float(sc))
                                    for it, sc in zip(inv_items, inv_scores)
                                )
                                sjt_sig_hist.append(sjt_sig)
                                inv_sig_hist.append(inv_sig)

                                sjt_run = 1
                                i = len(sjt_sig_hist) - 2
                                while i >= 0 and sjt_sig_hist[i] == sjt_sig_hist[-1]:
                                    sjt_run += 1
                                    i -= 1

                                inv_run = 1
                                i = len(inv_sig_hist) - 2
                                while i >= 0 and inv_sig_hist[i] == inv_sig_hist[-1]:
                                    inv_run += 1
                                    i -= 1

                                log_io(
                                    f"Verbatim run: SJT={sjt_run} INV={inv_run} "
                                    f"(cls={cls_s} group={group_index} alpha={alpha_factor:g})"
                                )

                                if len(sjt_sig_hist) >= 3:
                                    last3_sjt = sjt_sig_hist[-3:]
                                    sjt_same = True
                                    for x in last3_sjt[1:]:
                                        if x != last3_sjt[0]:
                                            sjt_same = False
                                            break

                                    if len(inv_sig_hist) >= 3:
                                        last3_inv = inv_sig_hist[-3:]
                                        inv_same = True
                                        for x in last3_inv[1:]:
                                            if x != last3_inv[0]:
                                                inv_same = False
                                                break
                                        if sjt_same and inv_same:
                                            log_io(
                                                f"Early stop (verbatim repetition, last 3): "
                                                f"cls={cls_s} group={group_index} layers={resolved_layers} alpha={alpha_factor:g}"
                                            )
                                            break

                                cutoff = float(CONFIG["fluency_outlier_threshold_factor"]) * float(baseline_avg_fluency)

                                mean_flu = 0.0
                                outliers = 0
                                for e in batch_entries:
                                    v = float(e["fluency"])
                                    mean_flu += v
                                    if v < cutoff:
                                        outliers += 1
                                mean_flu /= float(len(batch_entries))

                                stop_mean = mean_flu < (
                                    float(baseline_avg_fluency)
                                    * (1.0 - float(CONFIG["mean_fluency_delta_threshold"]))
                                )

                                allowed = int(math.ceil(float(CONFIG["fluency_model_uncertainty"]) * float(len(batch_entries))))
                                stop_outliers = outliers > allowed

                                if stop_mean or stop_outliers:
                                    break

                                step_index += 1

                    if not results:
                        continue

                    log_io(f"Embedding {len(results)} texts for probe scoring…")
                    texts = [r["text"] for r in results]

                    torch.cuda.empty_cache()
                    X_gen = embed_texts(
                        embed_tok,
                        embed_model,
                        texts,
                        int(CONFIG["probe_embed_batch"]),
                    )

                    log_io("Probe embeddings done. Running classifier…")
                    probs = probe_clf.predict_proba(X_gen)
                    classes = list(probe_clf.classes_)
                    pos_index = classes.index(1)
                    for i, r in enumerate(results):
                        r["logit"] = float(probs[i][pos_index])

                    log_io(f"Writing SQLite tables into sjts_responses.db ({len(results)} SJT rows)…")
                    write_sqlite(args_job, layer_groups, results)
                    log_io("SJT DB write done.")

                    log_io(f"Writing SQLite tables into inventory_responses.db ({len(inv_results)} inventory rows)…")
                    write_inventory_sqlite(args_job, layer_groups, inv_results)
                    log_io("Inventory DB write done.")


if __name__ == "__main__":
    main()
    print("########## ALL DONE ##########")
