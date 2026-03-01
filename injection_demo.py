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

system_text = "You are a person."
system_text = ""
prompt = "Write a short essay about Finding Nemo."
texts = inject_k_phase(
    model=model,
    tokenizer=tokenizer,
    method="meandiff",
    concepts=["feminine norms", "sadism", "masculine norms"],
    layers_per_concept=[[17], [11], [13]],
    alphas=[11, 7.4, 5.5],
    # alphas=[11.4, 7.8, 5.9],
    model_name=HF_MODEL_ID.split("/")[-1],
    max_new_tokens=450,
    batch_size=1,
    system_text=system_text,
    prompts=[prompt],
    mode="s",
    stride=1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
print(prompt)
print(texts[0][0])
print(texts[1])
