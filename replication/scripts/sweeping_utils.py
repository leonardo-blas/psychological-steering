import ast
import json
import sqlite3
from pathlib import Path
from decimal import Decimal
import joblib
from helpers import normalize_table_name
from injection_utils import VECTORS_ROOT, get_method_dir, get_mode_dir


CONFIG = {
    "seed": 42,
    "results_path": "results",
    "probe_embed_batch": 512,
    "probe_max_length": 64,
    "fluency_batch_size": 512,
    "sjts_db_path": "../data/sjts.db",
    "inventory_proper_names": {
        "mpi120": "MPI-120",
        "sd4": "SD4",
        "cmni30": "CMNI-30",
        "fmni45": "FMNI-45",
    },
    "fluency_model_uncertainty": 0.05,
    "mean_fluency_delta_threshold": 0.05,
    "fluency_outlier_threshold_factor": 0.9,
}


def classifier_path(concept: str) -> Path:
    root = Path("classifiers")
    root.mkdir(parents=True, exist_ok=True)
    concept_norm = normalize_table_name(concept)
    return root / f"{concept_norm}.pkl"


def load_classifier(concept: str):
    path = classifier_path(concept)
    if not path.is_file():
        raise FileNotFoundError(f"Classifier not found at {path}")
    return joblib.load(path)


def validate_args(args):
    start_spec = args.start is not None
    end_spec = args.end is not None

    if args.stride < 0:
        raise ValueError("--stride must be > 0.")

    if args.fluency_constrained:
        if start_spec or end_spec:
            raise ValueError("Cannot use --start/--end with -f/--fluency_constrained.")
    else:
        if start_spec != end_spec:
            raise ValueError("You must specify both --start and --end, or neither.")
        if not start_spec and not end_spec:
            args.start = 0.0
            args.end = 2.0

    if args.step <= 0.0:
        raise ValueError("--step must be positive.")

    args.framework = CONFIG["inventory_proper_names"].get(args.inventory, str(args.inventory))


def load_distances(model_name: str, concept: str, method: str, fit_intercept: bool, mode: str):
    model_short = model_name.split("/")[-1]
    concept_norm = normalize_table_name(concept)
    method_dir = "meandiff" if method == "meandiff" else get_method_dir(method, fit_intercept)
    mode_dir = get_mode_dir(mode)
    base = VECTORS_ROOT / model_short / concept_norm / method_dir / mode_dir
    path = base / "distances.json"
    if not path.is_file():
        raise FileNotFoundError(f"distances.json not found at {path}")
    with path.open("r") as f:
        return json.load(f)


def load_metrics(model_name: str, concept: str, method: str, fit_intercept: bool, mode: str):
    model_short = model_name.split("/")[-1]
    concept_norm = normalize_table_name(concept)
    method_dir = get_method_dir(method, fit_intercept)
    mode_dir = get_mode_dir(mode)
    base = VECTORS_ROOT / model_short / concept_norm / method_dir / mode_dir
    train_path = base / "train_metrics.json"
    test_path = base / "test_metrics.json"
    if not train_path.is_file():
        raise FileNotFoundError(f"train_metrics.json not found at {train_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"test_metrics.json not found at {test_path}")
    with train_path.open("r") as f:
        train = json.load(f)
    with test_path.open("r") as f:
        test = json.load(f)
    return train, test


def parse_layers_arg(raw: str, num_layers: int):
    try:
        value = ast.literal_eval(raw)
    except Exception as e:
        raise ValueError(f"--layers must be a Python-style list; got {raw!r}") from e
    if not isinstance(value, list):
        raise ValueError("--layers must be a list at top level.")

    def contains_minus_one(x):
        if isinstance(x, int):
            return x == -1
        if isinstance(x, list):
            for y in x:
                if contains_minus_one(y):
                    return True
            return False
        return False

    if contains_minus_one(value):
        all_layers = []
        for L in range(num_layers):
            all_layers.append(L)
        return [all_layers]

    layer_groups = []
    for entry in value:
        if isinstance(entry, int):
            group = [entry]
        elif isinstance(entry, list):
            if not entry:
                raise ValueError("Empty layer groups are not allowed.")
            for x in entry:
                if isinstance(x, list):
                    raise ValueError("Sub-sub lists in --layers are not allowed.")
                if not isinstance(x, int):
                    raise ValueError("Layer indices must be integers.")
            group = list(entry)
        else:
            raise ValueError("Each element of --layers must be an int or a list of ints.")

        uniq = []
        for L in group:
            if L < 0:
                raise ValueError("Negative layers (other than -1) are not allowed.")
            if L >= num_layers:
                raise ValueError(f"Layer {L} out of range for model with {num_layers} layers.")
            if L not in uniq:
                uniq.append(L)
        uniq.sort()
        layer_groups.append(uniq)

    if not layer_groups:
        raise ValueError("No layers provided.")
    return layer_groups


def compute_layers_suffix(layer_groups, num_layers: int) -> str:
    if len(layer_groups) == 1 and len(layer_groups[0]) == num_layers:
        return "_-1"
    all_layers = sorted({L for group in layer_groups for L in group})
    if not all_layers:
        return ""
    ranges = []
    start = all_layers[0]
    prev = all_layers[0]
    for L in all_layers[1:]:
        if L == prev + 1:
            prev = L
        else:
            ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = L
            prev = L
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return "_" + "_".join(ranges)


def filter_layer_groups_perfect(layer_groups, distances, train_metrics, test_metrics):
    good_layers = set()
    for L_str in distances.keys():
        t_layer = train_metrics.get(L_str)
        s_layer = test_metrics.get(L_str)
        if t_layer is None or s_layer is None:
            continue
        if t_layer.get("acc") == 1.0 and s_layer.get("acc") == 1.0:
            good_layers.add(int(L_str))
    new_groups = []
    for resolved_layers in layer_groups:
        kept = []
        for L in resolved_layers:
            if L in good_layers:
                kept.append(L)
        if kept:
            new_groups.append(kept)
    return new_groups


def format_layer_group(layers):
    if not layers:
        return ""
    layers_sorted = sorted(set(layers))
    segments = []
    start = layers_sorted[0]
    prev = layers_sorted[0]
    for L in layers_sorted[1:]:
        if L == prev + 1:
            prev = L
        else:
            segments.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = L
            prev = L
    segments.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ", ".join(segments)


def layer_group_label(layers):
    if not layers:
        return "empty"
    layers_sorted = sorted(set(layers))
    segments = []
    start = layers_sorted[0]
    prev = layers_sorted[0]
    for L in layers_sorted[1:]:
        if L == prev + 1:
            prev = L
        else:
            segments.append(f"{start}" if start == prev else f"{start}to{prev}")
            start = L
            prev = L
    segments.append(f"{start}" if start == prev else f"{start}to{prev}")
    return "_".join(segments)


def group_label(args, layers):
    base = layer_group_label(layers)
    tag = getattr(args, "tag", None)
    if tag:
        return f"{tag}_{base}"
    return base


def get_output_root(args):
    model_short = args.model.split("/")[-1]
    concept_norm = normalize_table_name(args.concept)
    method_dir = "meandiff" if args.method == "meandiff" else get_method_dir(args.method, args.fit_intercept)
    mode_dir = get_mode_dir(args.mode)
    stride_dir = f"stride_{args.stride}"
    step_dir = f"step_{args.step:g}"
    root = Path(CONFIG["results_path"]) / stride_dir / step_dir / model_short / concept_norm / method_dir / mode_dir
    root.mkdir(parents=True, exist_ok=True)
    return root


def intended_alpha(start: float, step: float, index: int) -> float:
    s = Decimal(str(start))
    st = Decimal(str(step))
    v = s + st * Decimal(index)
    return float(v)


def iter_alphas(start, end, step):
    if step <= 0.0:
        raise ValueError("--step must be positive.")
    if end < start:
        return []
    n_steps = int(round((end - start) / step))
    out = []
    for i in range(n_steps + 1):
        out.append(start + i * step)
    return out


def _alphas_string_for_row(args, class_val, alpha_factor):
    if class_val is None:
        return "[0]"
    combo_alphas = getattr(args, "combo_alphas", None)
    if combo_alphas is not None:
        return combo_alphas
    return f"[{float(alpha_factor):g}]"


def write_sqlite(args, layer_groups, results):
    out_root = get_output_root(args)
    responses_path = out_root / "sjts_responses.db"

    with sqlite3.connect(responses_path) as resp_conn:
        resp_cur = resp_conn.cursor()

        for group_index, layers_here in enumerate(layer_groups):
            table_name = group_label(args, layers_here)

            resp_cur.execute(
                f'CREATE TABLE IF NOT EXISTS "{table_name}" ('
                "class INTEGER, "
                "sjt TEXT, "
                "answer TEXT, "
                "alphas TEXT, "
                "concept_score REAL, "
                "fluency_score REAL)"
            )

            group_rows = []
            for r in results:
                if r["group"] == group_index:
                    group_rows.append(r)

            for r in group_rows:
                label = r.get("cls")
                class_val = 1 if label == "1" else 0 if label == "0" else None
                alphas_str = _alphas_string_for_row(args, class_val, r.get("alpha_factor", 0.0))
                resp_cur.execute(
                    f'INSERT INTO "{table_name}" '
                    "(class, sjt, answer, alphas, concept_score, fluency_score) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        class_val,
                        r.get("question", ""),
                        r["text"],
                        alphas_str,
                        float(r["logit"]),
                        float(r["fluency"]),
                    ),
                )

        resp_conn.commit()


def write_inventory_sqlite(args, layer_groups, inv_results):
    out_root = get_output_root(args)
    responses_path = out_root / "inventory_responses.db"

    with sqlite3.connect(responses_path) as resp_conn:
        resp_cur = resp_conn.cursor()

        for group_index, layers_here in enumerate(layer_groups):
            table_name = group_label(args, layers_here)

            resp_cur.execute(
                f'CREATE TABLE IF NOT EXISTS "{table_name}" ('
                "class INTEGER, "
                "item TEXT, "
                "score REAL, "
                "alphas TEXT)"
            )

            group_rows = []
            for r in inv_results:
                if r["group"] == group_index:
                    group_rows.append(r)

            for r in group_rows:
                label = r.get("cls")
                class_val = 1 if label == "1" else 0 if label == "0" else None
                alphas_str = _alphas_string_for_row(args, class_val, r.get("alpha_factor", 0.0))
                resp_cur.execute(
                    f'INSERT INTO "{table_name}" '
                    "(class, item, score, alphas) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        class_val,
                        r.get("item", ""),
                        float(r["score"]),
                        alphas_str,
                    ),
                )

        resp_conn.commit()
