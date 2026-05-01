from pathlib import Path
from typing import List
import torch
from tqdm.auto import tqdm
from helpers import normalize_table_name


VECTORS_ROOT = Path("vectors")


def get_inject_blocks(model, num_layers: int) -> torch.nn.ModuleList:
    blocks = getattr(model, "_inject_blocks", None)
    if blocks is not None:
        return blocks

    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "decoder", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    ]

    for b in candidates:
        if isinstance(b, torch.nn.ModuleList) and len(b) == int(num_layers):
            model._inject_blocks = b
            return b

    best = None
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) == int(num_layers):
            if name.endswith(("layers", "h", "blocks")):
                model._inject_blocks = m
                return m
            if best is None:
                best = m

    if best is None:
        raise AttributeError("Could not locate transformer block ModuleList for injection hooks.")

    model._inject_blocks = best
    return best


def get_method_dir(method: str, fit_intercept: bool) -> str:
    if method == "meandiff":
        return "meandiff"
    if method not in ("l1", "l2"):
        raise ValueError(f"Unknown method: {method}")
    if fit_intercept:
        return f"{method}_fitted_intercept"
    return f"{method}_zero_intercept"


def get_mode_dir(mode: str) -> str:
    if mode == "b":
        return "binary_choice"
    if mode == "s":
        return "statement"
    raise ValueError(f"Unknown mode: {mode}")


def get_vector_path(
    model_name: str,
    concept: str,
    layer: int,
    method: str,
    fit_intercept: bool,
    mode: str,
) -> Path:
    model_short = model_name.split("/")[-1]
    concept_norm = normalize_table_name(concept)
    method_dir = get_method_dir(method, fit_intercept)
    mode_dir = get_mode_dir(mode)
    base = VECTORS_ROOT / model_short / concept_norm / method_dir / mode_dir

    if method == "meandiff":
        p = base / f"layer_{layer}.pt"
        if not p.exists():
            raise FileNotFoundError(f"No vector file found for layer {layer} in {base}")
        return p

    candidates: List[Path] = []
    patterns = [
        f"layer_{layer}_C_*.pt",
        f"layer_{layer}.pt",
        f"layer_{layer}_*.pt",
    ]
    for pattern in patterns:
        for p in base.glob(pattern):
            if p.name.endswith("_wb.pt"):
                continue
            candidates.append(p)
    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(f"No vector file found for layer {layer} in {base}")
    return candidates[0]


def clean_layers(layers: List[int], num_layers: int) -> List[int]:
    if not layers:
        raise ValueError("No layers provided.")
    vals = list(layers)
    if len(vals) == 1 and vals[0] == -1:
        return list(range(num_layers))
    out: List[int] = []
    for L in vals:
        if L < 0:
            raise ValueError("Negative layers are not allowed (except -1 alone).")
        if L >= num_layers:
            raise ValueError(f"Layer {L} out of range for model with {num_layers} layers.")
        if L not in out:
            out.append(L)
    out.sort()
    return out


def inject(
    model,
    tokenizer,
    method: str,
    concepts: List[str],
    layers: List[int],
    model_name: str,
    alphas,
    max_new_tokens: int,
    batch_size: int,
    system_text: str,
    prompts: List[str],
    assistant_prefix: str | None = None,
    fit_intercept: bool | None = None,
    mode: str | None = None,
    stride: int = 1,
    **generate_kwargs,
) -> List[str]:
    if alphas is None:
        raise ValueError("alphas must be provided.")
    if len(concepts) != len(alphas):
        raise ValueError("alphas must have one row per concept.")
    if not isinstance(alphas[0], (list, tuple)):
        raise ValueError("alphas must be a list of lists with shape [n_concepts][n_layers].")
    if mode is None:
        raise ValueError("mode must be provided.")

    if method in ("l1", "l2"):
        if fit_intercept is None:
            raise ValueError("fit_intercept must be provided for LR methods (l1/l2).")
        fit_intercept_val = bool(fit_intercept)
    elif method == "meandiff":
        fit_intercept_val = False
    else:
        raise ValueError(f"Unknown method: {method}")

    num_layers = int(model.config.text_config.num_hidden_layers) if hasattr(model.config, "text_config") else int(model.config.num_hidden_layers)
    blocks = get_inject_blocks(model, num_layers)

    if batch_size < 1:
        batch_size = 1

    if not layers:
        raise ValueError("No layers provided.")
    for L in layers:
        if L < 0 or L >= num_layers:
            raise ValueError(f"Layer {L} out of range for model with {num_layers} layers.")

    layers = clean_layers(layers, num_layers)

    n_concepts = len(concepts)
    n_layers = len(layers)
    for row in alphas:
        if len(row) != n_layers:
            raise ValueError("Each alpha row must have one value per layer.")

    num_beams = int(generate_kwargs.get("num_beams", 1))
    num_return_sequences = int(generate_kwargs.get("num_return_sequences", 1))
    if num_beams != 1 or num_return_sequences != 1:
        raise ValueError("Injection hook assumes num_beams==1 and num_return_sequences==1.")

    outputs: List[str] = []
    total_prompts = len(prompts)
    if total_prompts == 0:
        return outputs

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    with tqdm(total=total_prompts, desc="inject prompts", disable=False) as pbar:
        for start in range(0, total_prompts, batch_size):
            end = min(start + batch_size, total_prompts)
            batch_prompts = prompts[start:end]

            messages_list: List[list[dict]] = []
            for p in batch_prompts:
                messages_list.append(
                    [
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": p},
                    ]
                )

            assistant_starts_unp: List[int] = []
            for msgs in messages_list:
                txt_no = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                ids_no = tokenizer(txt_no, return_tensors="pt").input_ids[0]
                assistant_starts_unp.append(ids_no.size(0))

            chat_texts: List[str] = []
            for msgs in messages_list:
                txt = tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                if assistant_prefix:
                    txt += " " + assistant_prefix
                chat_texts.append(txt)

            inputs = tokenizer(
                chat_texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(model.device)

            attn = inputs["attention_mask"]
            pad_left = (attn.size(1) - attn.sum(dim=1)).tolist()
            assistant_starts = [s + int(pl) for s, pl in zip(assistant_starts_unp, pad_left)]

            input_ids = inputs["input_ids"]
            B, T_in = input_ids.shape

            v_per_layer: dict[int, torch.Tensor] = {}
            for j, layer_idx in enumerate(layers):
                v_total: torch.Tensor | None = None
                for i, concept in enumerate(concepts):
                    alpha_ij = float(alphas[i][j])
                    if alpha_ij == 0.0:
                        continue
                    vec_path = get_vector_path(
                        model_name=model_name,
                        concept=concept,
                        layer=layer_idx,
                        method=method,
                        fit_intercept=fit_intercept_val,
                        mode=mode,
                    )
                    v = torch.load(vec_path, map_location="cpu") * alpha_ij
                    v_total = v.clone() if v_total is None else (v_total + v)
                if v_total is not None:
                    v_per_layer[layer_idx] = v_total

            offsets = [T_in - s for s in assistant_starts]
            gen_steps = [0] * B
            step_bump_layer = max(v_per_layer) if v_per_layer else None

            handles = []
            for layer_idx, v_total in v_per_layer.items():
                def hook(
                    module,
                    module_input,
                    module_output,
                    v=v_total,
                    starts=assistant_starts,
                    stride_val=stride,
                    offs=offsets,
                    steps=gen_steps,
                    bump_layer=step_bump_layer,
                    layer_id=layer_idx,
                    t_in=T_in,
                ):
                    hidden = module_output[0] if isinstance(module_output, tuple) else module_output
                    v_local = v.to(hidden.device, dtype=hidden.dtype)
                    B_local, T_local, _ = hidden.shape

                    if T_local != 1 and T_local != t_in:
                        raise RuntimeError(f"Unexpected forward length T_local={T_local} (expected 1 or {t_in}).")

                    if T_local == 1:
                        for b_local in range(B_local):
                            k = steps[b_local]
                            if (offs[b_local] + k) % stride_val == 0:
                                hidden[b_local, 0] = hidden[b_local, 0] + v_local
                        if bump_layer is not None and layer_id == bump_layer:
                            for b_local in range(B_local):
                                steps[b_local] = steps[b_local] + 1
                        if isinstance(module_output, tuple):
                            return (hidden,) + module_output[1:]
                        return hidden

                    for b_local in range(B_local):
                        s = starts[b_local]
                        if s < 0 or s >= T_local:
                            continue
                        for t in range(s, T_local):
                            if (t - s) % stride_val == 0:
                                hidden[b_local, t] = hidden[b_local, t] + v_local

                    if isinstance(module_output, tuple):
                        return (hidden,) + module_output[1:]
                    return hidden

                handles.append(blocks[layer_idx].register_forward_hook(hook))

            kwargs_local = dict(generate_kwargs)
            kwargs_local["max_new_tokens"] = max_new_tokens
            with torch.no_grad():
                out = model.generate(**inputs, **kwargs_local)

            for h in handles:
                h.remove()

            T_prompt = T_in
            for b in range(out.size(0)):
                gen_ids = out[b, T_prompt:]
                text = assistant_prefix + tokenizer.decode(
                    gen_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                outputs.append(text.strip())

            pbar.update(len(batch_prompts))

    return outputs
