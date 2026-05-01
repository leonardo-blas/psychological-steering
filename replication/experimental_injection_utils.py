from typing import List, Tuple, Dict, Any
import torch
from tqdm.auto import tqdm
from injection_utils import get_inject_blocks, clean_layers, get_vector_path


def inject_k_phase(
    model,
    tokenizer,
    method: str,
    concepts: List[str],                 # length K, phase 0 is used for prefill
    layers_per_concept: List[List[int]], # length K
    model_name: str,
    alphas: List[float],                 # length K (scalar per phase)
    max_new_tokens: int,
    batch_size: int,
    system_text: str,
    prompts: List[str],
    assistant_prefix: str | None = None,
    fit_intercept: bool | None = None,
    mode: str | None = None,
    stride: int = 1,
    **generate_kwargs,
) -> Tuple[List[str], List[List[Dict[str, str]]]]:
    if mode is None:
        raise ValueError("mode must be provided.")
    K = len(concepts)
    if K < 1:
        raise ValueError("concepts must be non-empty.")
    if len(layers_per_concept) != K:
        raise ValueError("layers_per_concept must match concepts length.")
    if len(alphas) != K:
        raise ValueError("alphas must match concepts length.")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1.")
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
    layers_clean: List[List[int]] = []
    all_layers_set: set[int] = set()
    for i in range(K):
        ph_layers = clean_layers(layers_per_concept[i], num_layers)
        layers_clean.append(ph_layers)
        for L in ph_layers:
            all_layers_set.add(L)
    all_layers: List[int] = sorted(all_layers_set)
    num_beams = int(generate_kwargs.get("num_beams", 1))
    num_return_sequences = int(generate_kwargs.get("num_return_sequences", 1))
    if num_beams != 1 or num_return_sequences != 1:
        raise ValueError("Injection hook assumes num_beams==1 and num_return_sequences==1.")
    if batch_size < 1:
        batch_size = 1

    def phase_of(k: int) -> int:
        ph = (k * K) // max_new_tokens
        if ph < 0:
            return 0
        if ph >= K:
            return K - 1
        return ph
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    outputs: List[str] = []
    outputs_spans: List[List[Dict[str, str]]] = []
    if not prompts:
        return outputs, outputs_spans
    total_prompts = len(prompts)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
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
            v_per_layer_phase: dict[int, List[torch.Tensor | None]] = {}
            for layer_idx in all_layers:
                phase_vs: List[torch.Tensor | None] = [None] * K
                for ph in range(K):
                    if layer_idx not in layers_clean[ph]:
                        continue
                    alpha = float(alphas[ph])
                    if alpha == 0.0:
                        continue
                    vec_path = get_vector_path(
                        model_name=model_name,
                        concept=concepts[ph],
                        layer=layer_idx,
                        method=method,
                        fit_intercept=fit_intercept_val,
                        mode=mode,
                    )
                    phase_vs[ph] = torch.load(vec_path, map_location="cpu") * alpha
                if any(v is not None for v in phase_vs):
                    v_per_layer_phase[layer_idx] = phase_vs
            offsets = [T_in - s for s in assistant_starts]
            gen_steps = [0] * B
            step_bump_layer = max(v_per_layer_phase) if v_per_layer_phase else None
            handles = []
            for layer_idx, phase_vs in v_per_layer_phase.items():
                def hook(
                    module,
                    module_input,
                    module_output,
                    v_phase=phase_vs,
                    starts=assistant_starts,
                    stride_val=stride,
                    offs=offsets,
                    steps=gen_steps,
                    bump_layer=step_bump_layer,
                    layer_id=layer_idx,
                    t_in=T_in,
                ):
                    hidden = module_output[0] if isinstance(module_output, tuple) else module_output
                    B_local, T_local, _ = hidden.shape
                    if T_local != 1 and T_local != t_in:
                        raise RuntimeError(f"Unexpected forward length T_local={T_local} (expected 1 or {t_in}).")
                    if T_local == t_in:
                        v0 = v_phase[0]
                        if v0 is not None:
                            v_local = v0.to(hidden.device, dtype=hidden.dtype)
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
                    for b_local in range(B_local):
                        k = steps[b_local]
                        if (offs[b_local] + k) % stride_val != 0:
                            continue
                        ph = phase_of(k)
                        v = v_phase[ph]
                        if v is None:
                            continue
                        hidden[b_local, 0] = hidden[b_local, 0] + v.to(hidden.device, dtype=hidden.dtype)
                    if bump_layer is not None and layer_id == bump_layer:
                        for b_local in range(B_local):
                            steps[b_local] += 1
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
            for b in range(out.size(0)):
                gen_ids = out[b, T_in:].tolist()
                spans: List[Dict[str, str]] = []
                cur_concept: str | None = None
                cur_text = ""
                if assistant_prefix:
                    cur_concept = concepts[phase_of(0)]
                    cur_text = assistant_prefix
                for k, tid in enumerate(gen_ids):
                    if tid in special_ids:
                        continue
                    piece = tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    concept = concepts[phase_of(k)]
                    if concept != cur_concept:
                        if cur_concept is not None and cur_text:
                            spans.append({"concept": cur_concept, "text": cur_text})
                        cur_concept = concept
                        cur_text = piece
                    else:
                        cur_text += piece
                if cur_concept is not None and cur_text:
                    spans.append({"concept": cur_concept, "text": cur_text})
                outputs_spans.append(spans)
                text = (assistant_prefix or "") + tokenizer.decode(out[b, T_in:], skip_special_tokens=True)
                outputs.append(text.strip())
            pbar.update(len(batch_prompts))
    return outputs, outputs_spans
