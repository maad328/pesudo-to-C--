#!/usr/bin/env python3
"""
GPT-2  →  C++   (QLoRA + Trainer API)
- 4-bit quant → fits MAX_LEN=256 on 16 GB GPU (approx)
- saves best-BLEU adapter to ./gpt2-lora-cpp
- fast eval (200 samples, every epoch)
- generation-based BLEU / syntax check
"""
# ------------------------------------------------------------------
# 1.  installs (uncomment first run)
# ------------------------------------------------------------------
# !pip install -q transformers datasets peft accelerate sacrebleu evaluate bitsandbytes

import os
import json
import random
import subprocess
import tempfile
import shutil
import glob
from typing import List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
import sacrebleu

# ------------------------------------------------------------------
# reproducibility / env
# ------------------------------------------------------------------
set_seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["WANDB_DISABLED"] = "true"

# ------------------------------------------------------------------
# 2.  load TSV → HF dataset
# ------------------------------------------------------------------
def read_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert {"text", "code"}.issubset(df.columns)
    return df[["text", "code"]].replace("", pd.NA).dropna().reset_index(drop=True)

TRAIN_TSV = "/kaggle/input/code-dataset/spoc-train-train.tsv"
VAL_TSV   = "/kaggle/input/code-dataset/spoc-train-eval.tsv"

train_df = read_tsv(TRAIN_TSV)
train_df = train_df.sample(frac=0.7, random_state=42).reset_index(drop=True)

val_df   = read_tsv(VAL_TSV)

def make_example(pseudo: str, code: str) -> str:
    return f"### PSEUDO-CODE:\n{pseudo.strip()}\n### C++:\n{code.strip()}\n"

train_ds = Dataset.from_dict({
    "pseudo": train_df["text"].tolist(),
    "code"  : train_df["code"].tolist(),
})
val_ds   = Dataset.from_dict({
    "pseudo": val_df["text"].tolist(),
    "code"  : val_df["code"].tolist(),
})

# fast eval subset (random 200)
if len(val_ds) >= 50:
    val_ds = val_ds.select(random.sample(range(len(val_ds)), 50))

# ------------------------------------------------------------------
# 3.  QLoRA base model (4-bit quant)
# ------------------------------------------------------------------
model_name = "gpt2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# load tokenizer and set pad BEFORE any tokenization / collators
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# ensure we have a pad token; use eos as pad (common for decoder-only)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"        # <<< crucial: left padding for decoder-only models

# propagate pad/eos ids to base model BEFORE PEFT wrap
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.eos_token_id = tokenizer.eos_token_id

# create data collator AFTER tokenizer exists
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,            # causal LM
)

# PEFT LoRA config + wrap
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

# ensure model.config knows pad/eos (redundant but safe)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

# ------------------------------------------------------------------
# 4.  tokenise  (now left-padded)
# ------------------------------------------------------------------
MAX_LEN = 128

def tokenize(batch):
    texts = [make_example(p, c) for p, c in zip(batch["pseudo"], batch["code"])]
    tok = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding="max_length")
    tok["labels"] = tok["input_ids"].copy()
    return tok

tokenized_train = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
tokenized_val   = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)

# Ensure DataLoader works
tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ------------------------------------------------------------------
# 5.  C++ syntax checker
# ------------------------------------------------------------------
def cpp_ok(code: str, timeout=2) -> bool:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        return code.count("{") == code.count("}") and ";" in code
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False) as f:
            f.write(code); tmp = f.name
        rc = subprocess.run([compiler, "-fsyntax-only", tmp],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=timeout).returncode
        return rc == 0
    except Exception:
        return False
    finally:
        if tmp:
            try: os.remove(tmp)
            except: pass

# ------------------------------------------------------------------
# 6.  metrics: BLEU + syntax pass-rate  (generation-based)
# ------------------------------------------------------------------
def extract_code(s: str) -> str:
    if "### C++:" in s:
        return s.split("### C++:")[-1].strip()
    return s.strip()

def compute_metrics(eval_preds):
    # NOTE: trainer passes predictions and labels; we ignore eval_preds here because we do generation-based metrics
    hyps: List[str] = []
    refs: List[List[str]] = []

    dataloader = DataLoader(tokenized_val, batch_size=training_args.per_device_eval_batch_size)
    for batch in tqdm(dataloader, desc="Generate"):
        # move to device
        batch = {k: v.to(model.device) for k, v in batch.items()}

        # compute prompt length per sample (works with left padding)
        prompt_lens = (batch["input_ids"] != tokenizer.pad_token_id).sum(dim=1).tolist()

        # generate per sample (safe, slower)
        for i, plen in enumerate(prompt_lens):
            # take last `plen` tokens (left padding -> actual tokens are at the right of sequence)
            input_ids = batch["input_ids"][i, -plen:].unsqueeze(0)
            attention_mask = batch["attention_mask"][i, -plen:].unsqueeze(0)

            max_new = MAX_LEN - plen
            if max_new <= 0:
                # prompt fills MAX_LEN already; generate 1 token to avoid zero-length generation
                max_new = 1

            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            decoded = tokenizer.decode(out[0], skip_special_tokens=True)
            hyps.append(extract_code(decoded))

            # reference: decode labels for this sample
            ref = tokenizer.decode(batch["labels"][i], skip_special_tokens=True)
            refs.append([extract_code(ref)])

    # If nothing was generated, return dummy scores
    if len(hyps) == 0:
        return {"eval_bleu": 0.0, "eval_cpp_syntax": 0.0}

    bleu = sacrebleu.corpus_bleu(hyps, refs).score
    syn  = float(np.mean([cpp_ok(h) for h in hyps]) * 100.0)
    return {"eval_bleu": float(bleu), "eval_cpp_syntax": syn}

# ------------------------------------------------------------------
# 7.  TrainingArguments  (fast + best-BLEU save)
# ------------------------------------------------------------------
output_dir = "./gpt2-lora-cpp"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    dataloader_num_workers=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    save_total_limit=2,
    report_to="none",
    warmup_steps=50,
    weight_decay=0.01,

    load_best_model_at_end=True,
    metric_for_best_model="eval_bleu",
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ------------------------------------------------------------------
# 8.  train & save
# ------------------------------------------------------------------
trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("Best-BLEU adapter saved to", output_dir)
# -------------------------------------------------------------
# 9.  Manual Inference / Test Examples after training
# -------------------------------------------------------------
print("\n=== Running Manual Test Examples ===\n")

def generate_cpp(pseudo: str, max_new_tokens=150):
    prompt = f"### PSEUDO-CODE:\n{pseudo}\n### C++:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,   # deterministic output
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    if "### C++:" in decoded:
        decoded = decoded.split("### C++:")[1].strip()
    return decoded





