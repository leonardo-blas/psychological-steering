import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from experimental_injection_utils import inject_k_phase
from injection_utils import inject


HF_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, use_fast=True, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

system_text = "You are a person."  # Other system prompts work as well.
prompt = "Write a short essay about Finding Nemo."
texts = inject_k_phase(
    model=model,
    tokenizer=tokenizer,
    method="meandiff",
    concepts=["feminine norms", "sadism", "masculine norms"],
    layers_per_concept=[[17], [11], [13]],  # The best-performing layers.
    # alphas=[11.4, 7.8, 5.9],  # The best-performing coefficients.
    alphas=[10, 6.5, 5],
    stride=1,  # Inject on every token. If stride=2, it injects on every other token.
    model_name=HF_MODEL_ID.split("/")[-1],
    max_new_tokens=450,
    batch_size=1,
    system_text=system_text,
    prompts=[prompt],
    mode="s",
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
print(prompt)
print(texts[0][0])
# print(texts[1])  # This prints the set of tokens alonside the corresponding steering construct.
