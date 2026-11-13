
"""
LoRA + DPO fine-tuning of Qwen2.5-1.5B-Instruct on nvidia/When2Call
Target: Google Colab GPU (T4 works; L4/A100 also work).

Setup (run in its own cell first; do NOT reinstall torch, Colab's is fine):

%pip install -q --no-input -U transformers trl peft accelerate datasets

Runtime > Change runtime type > T4 GPU, then run this script.
"""

import json
import os
import time

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUT_DIR = "/kaggle/working/qwen2.5-1.5b-when2call-dpo"
SMOKE_TEST = False            # True = 20 optimizer steps, no eval/checkpointing
WARMUP_STEPS = 2 if SMOKE_TEST else 100
MAX_LEN_CAP = 1024           # lower to 768 or 512 if you hit out-of-memory

assert torch.cuda.is_available(), "No GPU. Use Runtime > Change runtime type > T4 GPU."

# T4 (compute capability 7.5) has no real bf16 support -> use fp16 there.
# Ampere or newer (L4, A100, ...) -> bf16.
major, _ = torch.cuda.get_device_capability()
USE_BF16 = major >= 8
compute_dtype = torch.bfloat16 if USE_BF16 else torch.float16
print(f"GPU: {torch.cuda.get_device_name(0)} | compute dtype: {compute_dtype}")

# --------------------------------------------------------------------------
# Tokenizer and model
# --------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def load_model():
    kwargs = dict(attn_implementation="sdpa", device_map={"": 0})
    try:  # transformers >= 4.56
        return AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=compute_dtype, **kwargs)
    except TypeError:  # older transformers
        return AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=compute_dtype, **kwargs)


model = load_model()
model.config.use_cache = False

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
# PEFT keeps the LoRA weights in float32 while the base model stays fp16/bf16.
# This is required for fp16 training (the GradScaler can't unscale fp16 params).
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
raw_dataset = load_dataset("nvidia/When2Call", "train_pref", split="train")


def _to_text(resp):
    return json.dumps(resp) if isinstance(resp, dict) else resp


def format_when2call(sample):
    tools = []
    for t in sample.get("tools") or []:
        if isinstance(t, str):
            try:
                tools.append(json.loads(t))
            except json.JSONDecodeError:
                pass
        elif isinstance(t, dict):
            tools.append(t)

    prompt_str = tokenizer.apply_chat_template(
        sample["messages"],
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=True,
    )
    # TRL appends EOS itself, so don't add it here (it would be doubled).
    return {
        "prompt": prompt_str,
        "chosen": _to_text(sample["chosen_response"]),
        "rejected": _to_text(sample["rejected_response"]),
    }


dpo_dataset = raw_dataset.map(
    format_when2call,
    remove_columns=raw_dataset.column_names,
    num_proc=4,
)


def add_len(ex):
    longer = max(ex["chosen"], ex["rejected"], key=len)
    n = len(tokenizer(ex["prompt"] + longer, add_special_tokens=False).input_ids) + 1
    return {"n_tokens": n}


dpo_dataset = dpo_dataset.map(add_len, num_proc=4)

p95, p99 = np.percentile(dpo_dataset["n_tokens"], [95, 99])
MAX_LEN = int(min(MAX_LEN_CAP, -(-p99 // 64) * 64))
print(f"token length p95={p95:.0f} p99={p99:.0f} -> using max_length={MAX_LEN}")

# Drop the few examples that don't fit instead of truncating them.
before = len(dpo_dataset)
dpo_dataset = dpo_dataset.filter(lambda ex: ex["n_tokens"] <= MAX_LEN)
dpo_dataset = dpo_dataset.remove_columns(["n_tokens"])
print(f"kept {len(dpo_dataset)}/{before} examples (<= {MAX_LEN} tokens)")

splits = dpo_dataset.train_test_split(test_size=200, seed=42)
train_ds, eval_ds = splits["train"], splits["test"]

# --------------------------------------------------------------------------
# Training config
# --------------------------------------------------------------------------
cfg = dict(
    output_dir=OUT_DIR,
    report_to="none",
    seed=42,

    beta=0.1,
    precompute_ref_log_probs=False,
    max_length=MAX_LEN,

    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,
    num_train_epochs=1,
    max_steps=20 if SMOKE_TEST else -1,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    optim="adamw_torch",
    fp16=not USE_BF16,
    bf16=USE_BF16,

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    logging_first_step=True,
    logging_steps=1 if SMOKE_TEST else 10,

    eval_strategy="no" if SMOKE_TEST else "steps",
    eval_steps=200,
    per_device_eval_batch_size=1,

    save_strategy="no" if SMOKE_TEST else "steps",
    save_steps=100,
    save_total_limit=2,

    dataloader_num_workers=0,
    remove_unused_columns=False,
    push_to_hub=False,
)

# Older TRL versions default max_prompt_length to 512 and would truncate prompts.
if "max_prompt_length" in DPOConfig.__dataclass_fields__:
    cfg["max_prompt_length"] = MAX_LEN

training_args = DPOConfig(**cfg)

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # with a PEFT model, the reference = adapter disabled
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
)


class ProgressPrinter(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        self.s0 = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        done = state.global_step - self.s0
        if done > 0:
            per = (time.time() - self.t0) / done
            eta_h = per * (state.max_steps - state.global_step) / 3600
            print(
                f"[step {state.global_step}/{state.max_steps}] "
                f"{per:.1f}s/step  ETA {eta_h:.2f}h",
                flush=True,
            )


trainer.add_callback(ProgressPrinter())

# --------------------------------------------------------------------------
# Train and save
# --------------------------------------------------------------------------
resume = (
    not SMOKE_TEST
    and os.path.isdir(OUT_DIR)
    and any(d.startswith("checkpoint-") for d in os.listdir(OUT_DIR))
)
trainer.train(resume_from_checkpoint=resume)

trainer.save_model(f"{OUT_DIR}/final_adapter")
tokenizer.save_pretrained(f"{OUT_DIR}/final_adapter")
print("Saved adapter to", f"{OUT_DIR}/final_adapter")
