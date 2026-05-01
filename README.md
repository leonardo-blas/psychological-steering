# Psychological Steering of Large Language Models

```
@misc{blas2026psychologicalsteeringlargelanguage,
      title={Psychological Steering of Large Language Models}, 
      author={Leonardo Blas and Robin Jia and Emilio Ferrara},
      year={2026},
      eprint={2604.14463},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.14463}, 
}
```

<img width="1631" height="516" alt="Screenshot 2026-04-30 at 2 59 17 PM" src="https://github.com/user-attachments/assets/36c36cb4-8cb1-436f-878e-b833983b34d6" />

Figure 1: Llama-3.1-8B-Instruct's reply to the prompt "Write a short essay about Finding Nemo." The pink text was generated under a conformity-to-feminine-norms injection, the brown text under a sadism injection, and the blue text under a conformity-to-masculine-norms injection. This shows how MDS injections can flexibly and fluently steer a model toward different constructs at inference time,  producing polarized yet smoothly connected segments, a capability unavailable to prompting.


# Replication instructions

The first step is to install all requirements:

```
conda create -n psych_steering python=3.12.12
conda activate psych_steering
pip install -r requirements.txt
```

## To replicate Figure 1

```
python3 injection_demo.py
```

Modify the script to achieve different steering results or pass different instructions.

## For full replication

1. First, we synthesize statements expressing a construct or its antithesis.
   - `python3 1_create_statements.py --concept "neuroticism" --phrase " is neurotic."`
2. Second, we apply a fluency filter and semantically deduplicate the raw statements.
   - `python3 2_filter_statements.py --concept "neuroticism" --phrase " is neurotic."`
3. Third, we extract and save layerwise activations locally, in `data`. Be careful, they can be very heavy. This is not optimal but helped train L1-regularized probes. `--mode` is the type of activations; `s` means statement and `b` means binary. See the paper for in-depth explanations. Quantization is avaiable with `-q`.
   - `python3 3_get_activations.py --concept "neuroticism" --model Qwen/Qwen3-1.7B --mode s`
   - `python3 3_get_activations.py --concept "neuroticism" --model Qwen/Qwen3-1.7B --mode b`
4. Fourth, we create mean-difference and probe-based vectors. `-i` means intercept fit and `-r` is the regularization type. Probe-based vectors were created only with `b` activations.
   - `python3 4_create_vectors_lr.py --concept "neuroticism" --model Qwen/Qwen3-1.7B -r l1 -i --mode b`
   - `python3 4_create_vectors_lr.py --concept "neuroticism"" --model Qwen/Qwen3-1.7B -r l1 --mode b`
   - `python3 4_create_vectors_lr.py --concept "neuroticism" --model Qwen/Qwen3-1.7B -r l2 -i --mode b`
   - `python3 4_create_vectors_lr.py --concept "neuroticism" --model Qwen/Qwen3-1.7B -r l2 --mode b`
   - `python3 4_create_vectors_meandiff.py --concept "neuroticism" --model Qwen/Qwen3-1.7B --mode b`
   - `python3 4_create_vectors_meandiff.py --concept "neuroticism" --model Qwen/Qwen3-1.7B --mode s`
5. Fifth, we filter ATOMIC10X, which is needed to create SJTs. The filtered version was too large to be included here. To get the raw version, visit https://github.com/peterwestai2/symbolic-knowledge-distillation. This step needs `data/ATOMIC10X.jsonl` and produces `data/heads.db`.
   - `python3 5_filter_atomic10x.py`
6. Sixth, we create SJTs. This automatically creates 25 SJTs per inventory item, for each inventory in `data/inventories.db`. This needs `data/heads.db` and your `OPENAI_API_KEY` as an environment variable. In the paper and in this repository, we use the MPI-120 (a second-person version of the IPIP-NEO-120). Note that the IPIP-NE0-120 is designed to measure OCEAN constructs such as the one we are targeting, neuroticism.
   - `export OPENAI_API_KEY="your_api_key_here"`
   - `6_create_sjts.py`
7. Seventh, we we apply a fluency filter and semantically deduplicate the raw SJTs.
   - `7_filter_sjts.py`
8. Eight, we train the classifiers based on the concepts we have created and filtered statements for.
   - `8_train_classifiers.py`
9. Ninth, we sweep our vectors. Quantization is avaiable with `-q`.
    - `python3 9_sweep_injection_alphas.py --model Qwen/Qwen3-1.7B --inventory mpi120 --mode b --batch_size 128 --method l1_and_l2 -i --stride 1`
    - `python3 9_sweep_injection_alphas.py --model Qwen/Qwen3-1.7B --inventory mpi120 --mode b --batch_size 128 --method l1 -i -q --stride 1`
    - `python3 9_sweep_injection_alphas.py --model Qwen/Qwen3-1.7B --inventory mpi120 --mode b --batch_size 128 --method l1 -q --stride 1`
    - `python3 9_sweep_injection_alphas.py --model Qwen/Qwen3-1.7B --inventory mpi120 --mode b --batch_size 128 --method l2 -i -q --stride 1`
    - `python3 9_sweep_injection_alphas.py --model Qwen/Qwen3-1.7B --inventory mpi120 --mode b --batch_size 128 --method l2 -q --stride 1`
    - `python3 9_sweep_injection_alphas.py --model Qwen/Qwen3-1.7B --inventory mpi120 --mode s --batch_size 128 --method meandiff --stride 1`
    - `python3 9_sweep_injection_alphas.py --model Qwen/Qwen3-1.7B --inventory mpi120 --mode b --batch_size 128 --method meandiff --stride 1`
10. Tenth, we get $P^2$ OCEAN baseline results. Quantization is avaiable with `-q`.
    - `10_get_p2_ocean_baseline.py --model Qwen/Qwen3-1.7B --batch_size 128`
11. Lastly, we conduct OCEAN cross-trait sweeps using only MDS (mean-difference, statement activations) injections. Quantization is avaiable with `-q`.
    - `11_cross_trait_sweeps.py --model Qwen/Qwen3-1.7B --batch_size 128`
   
## After getting sweep results
To get the best MDS injection settings (layer and coefficient) for a concept, do:

`python3 get_best_intervention_settings.py --concept "openness" --model Qwen/Qwen3-1.7B`

To inject an MDS vector, do:
```py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from injection_utils import inject

HF_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.clean_up_tokenization_spaces = False

model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
).eval()

texts = inject(
    model=model,
    tokenizer=tokenizer,
    method="meandiff",
    concepts=["openness"],
    layers=[12],
    model_name=HF_MODEL_ID,
    alphas=[[-5]],
    mode="s",
    stride=1,
    max_new_tokens=128,
    batch_size=1,
    system_text="You are a zoologist.",
    prompts=["Do you like alpacas?"],
    do_sample=True,
    temperature=0.5,
    top_p=0.9,
    repetition_penalty=1.1,
    assistant_prefix="I",
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

print(texts[0])

```
