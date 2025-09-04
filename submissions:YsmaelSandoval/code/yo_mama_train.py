"""
Train a small GPT-style model on 'yo mama' jokes, then generate new ones.

Data source (default): Fraser/short-jokes on Hugging Face (231k jokes; id + text).
We filter to jokes that contain 'yo mama|yo momma|your mom' and drop offensive ones.

After training, try generation with a 'Yo mama' prefix.

"""

import os, re, math, random
from typing import List
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments, Trainer, pipeline, set_seed
)
import numpy as np

DATASET = "ysharma/short_jokes"    # alt datasets listed in notes at bottom
TEXT_KEYS = ["Joke", "joke", "text", "body", "content", "title"]
YOMAMA_REGEX = re.compile(r"\b(yo ?mama|yo ?momma|your mom)\b", re.I)
BLOCKLIST = [
    # crude/offensive filters — add more terms you don't want to train on
    "slur", "racist", "nazi", "heil", "lynch", "rape", "pedoph", "kill yourself"
]
MODEL_NAME = "gpt2"          # small; runs on CPU (slow but works), GPU if available is better
OUTPUT_DIR = "yo-mama-gpt"
MAX_TOKENS_PER_EXAMPLE = 64        # keep training examples short
TRAIN_SAMPLES_CAP = 30000           # cap for speed; set None to use all matches
VAL_RATIO = 0.05                   # 5% validation
BATCH_TRAIN = 8
BATCH_EVAL = 8
EPOCHS = 5
LR = 2e-5
SEED = 42
GEN_MAX_NEW_TOKENS = 40
GEN_NUM_RETURNS = 6
# --------------------------------------------------------------------

def get_text(row):
    for k in TEXT_KEYS:
        if k in row and isinstance(row[k], str) and row[k].strip():
            return row[k].strip()
    return None

def safe(row_text: str) -> bool:
    t = row_text.lower()
    return not any(b in t for b in BLOCKLIST)

def main():
    set_seed(SEED)

    # 1) Load dataset from the Hub
    # Fraser/short-jokes: “id + joke” (CSV) with 231,657 jokes. (We’ll robustly grab text.)
    ds = load_dataset(DATASET, split="train")  # single split dataset
    print(ds)

    # 2) Extract text, filter to yo-mama jokes, apply basic safety filter
    ds = ds.map(lambda r: {"text": get_text(r)})
    ds = ds.filter(lambda r: r["text"] is not None)
    print("Raw rows with any text:", len(ds))

    ds = ds.filter(lambda r: bool(YOMAMA_REGEX.search(r["text"])))
    print("Rows matching yo-mama regex:", len(ds))

    ds = ds.filter(lambda r: safe(r["text"]))
    print("Rows after blocklist filter:", len(ds))

    # Optional: dedupe (case-insensitive)
    seen = set()
    def dedupe(example):
        key = example["text"].strip().lower()
        if key in seen:
            return False
        seen.add(key)
        return True
    ds = ds.filter(dedupe)
    print("Rows after dedupe:", len(ds))

    if TRAIN_SAMPLES_CAP:
        ds = ds.shuffle(seed=SEED).select(range(min(TRAIN_SAMPLES_CAP, len(ds))))
        print("Capped to:", len(ds))

    # Add an explicit prefix to encourage style & format
    # (helps the model learn to start with “Yo mama ...”)
    def add_prefix(r):
        txt = r["text"]
        # Ensure it starts with a consistent cue
        if not txt.lower().startswith(("yo mama", "yo momma", "your mom")):
            txt = "Yo mama " + txt
        return {"text": txt}
    ds = ds.map(add_prefix)

    # 3) Train/val split
    ds = ds.train_test_split(test_size=VAL_RATIO, seed=SEED)
    train_ds, val_ds = ds["train"], ds["test"]
    print("Train size:", len(train_ds), "Val size:", len(val_ds))

    # 4) Tokenizer & model
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    # GPT-2 family has no pad token; set pad to eos for batching
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=MAX_TOKENS_PER_EXAMPLE)

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
    val_tok   = val_ds.map(tokenize,   batched=True, remove_columns=val_ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tok))  # safe even when unchanged

    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # 5) Training setup
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",  # <-- new name
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
        seed=SEED,
    )

    def compute_metrics(eval_pred):
        return {}

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    # 6) Train & evaluate
    trainer.train()
    eval_out = trainer.evaluate()
    val_loss = float(eval_out.get("eval_loss", np.nan))
    ppl = math.exp(val_loss) if val_loss and val_loss < 20 else float("inf")
    print({"eval_loss": round(val_loss, 4), "perplexity": round(ppl, 2)})

    # 7) Save
    best_dir = os.path.join(OUTPUT_DIR, "final")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)
    tok.save_pretrained(best_dir)
    print("Saved to:", best_dir)

    # 8) Sample some jokes
    # 8) Sample some jokes (shorter + cleaner)
    from transformers import pipeline
    import re

    gen = pipeline("text-generation", model=trainer.model, tokenizer=tok)

    def one_liner(
            prompt,
            max_new=12,  # shorter output (try 12–18)
            temperature=.3,  # cooler = less random
            top_p=0.8,
            top_k=50,
            repetition_penalty=1.3
    ):
        out = gen(
            prompt,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            num_return_sequences=1,
        )[0]["generated_text"]

        # keep only the first sentence after your prompt
        completion = re.split(r'(?<=[.!?])\s', out[len(prompt):].strip())[0]
        return prompt + completion

    prompts = [
        "Yo mama so clumsy,",
        "Yo mama so smart,",
        "Yo momma so old,",
        "Your mom is so fast,",
    ]

    print("\n=== Samples ===")
    for p in prompts:
        print("-", one_liner(p))


if __name__ == "__main__":
    main()
