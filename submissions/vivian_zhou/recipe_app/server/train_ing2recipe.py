import os, re, json
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, TaskType

MODEL = "google/flan-t5-small"
OUT_DIR = "models/ing2recipe_lora"
MAX_N = 2500
MAX_IN, MAX_OUT = 128, 192

print("Loading dataset…")
ds = load_dataset("AkashPS11/recipes_data_food.com")
raw = ds["train"]
print("Columns:", raw.column_names)
print("Sample[0] keys:", list(raw[0].keys()))

# ---- Helpers ----
def pick_field(ex, keys):
    for k in keys:
        if k in ex:
            v = ex[k]
            if v not in (None, "", [], {}):
                return v
    return None

def normalize_ings(v):
    if isinstance(v, list):
        return [str(i).strip() for i in v if str(i).strip()]
    if isinstance(v, str):
        parts = re.split(r"[\n,;]+", v)
        return [p.strip() for p in parts if p.strip()]
    return []

def normalize_steps(v):
    if isinstance(v, list):
        return [str(s).strip() for s in v if str(s).strip()]
    if isinstance(v, str):
        return [s.strip() for s in re.split(r"[.\n;]+", v) if s.strip()]
    return []

# ---- Field keys (corrected) ----
TITLE_KEYS = ["Name", "RecipeId", "AuthorName"]          # dataset usually has "Name"
ING_KEYS   = ["RecipeIngredientParts", "RecipeCategory", "RecipeIngredientQuantities"]  # fixed typo
STEP_KEYS  = ["RecipeInstructions", "RecipeYield", "RecipeServings"]

# ---- Filter to rows that actually have data ----
def has_min_fields(ex):
    return bool(pick_field(ex, TITLE_KEYS)) and bool(pick_field(ex, ING_KEYS)) and bool(pick_field(ex, STEP_KEYS))

filtered = raw.filter(has_min_fields)
print("After filter:", len(filtered))
assert len(filtered) > 0, "No examples after filter—inspect columns/keys."

# ---- Subsample + split FROM FILTERED (not raw) ----
N = min(MAX_N, len(filtered))
filtered = filtered.shuffle(seed=42).select(range(N))
splits = filtered.train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = splits["train"], splits["test"]
print("Train/Val sizes:", len(train_ds), len(val_ds))

# ---- Map to (src, tgt) safely ----
def to_io(ex):
    title = str(pick_field(ex, TITLE_KEYS))[:60]
    ings  = normalize_ings(pick_field(ex, ING_KEYS))[:20]
    steps = normalize_steps(pick_field(ex, STEP_KEYS))[:7]

    # Guard against accidental Nones/empties
    if not ings or not steps:
        # return a minimal-but-valid pair so map() doesn't crash, you can also choose to drop row here
        ings, steps = ings or ["salt", "pepper"], steps or ["Combine and serve."]

    src = (
        "You are a professional chef. Create a concise recipe that ONLY uses the provided pantry "
        "items plus salt, pepper, oil, and water. 5–8 numbered steps.\n"
        f"Pantry: {', '.join(ings)}\n"
        "Return JSON with keys: Name, RecipeIngredientParts, RecipeInstructions."
    )
    tgt = json.dumps({
        "Name": title,
        "RecipeIngredientParts": ings[:12],
        "RecipeInstructions": steps
    }, ensure_ascii=False)
    return {"src": src, "tgt": tgt}

train_io = train_ds.map(to_io, remove_columns=train_ds.column_names)
val_io   = val_ds.map(to_io,   remove_columns=val_ds.column_names)
print("IO sizes:", len(train_io), len(val_io))

# ---- Tokenize; prefer text_target=... on newer transformers ----
tok = AutoTokenizer.from_pretrained(MODEL)

def tok_fn(batch):
    X = tok(batch["src"], max_length=MAX_IN, truncation=True)
    Y = tok(text_target=batch["tgt"], max_length=MAX_OUT, truncation=True)
    X["labels"] = Y["input_ids"]
    return X

enc_train = train_io.map(tok_fn, batched=True, remove_columns=train_io.column_names)
enc_val   = val_io.map(tok_fn,   batched=True, remove_columns=val_io.column_names)
print("Encoded train/val:", len(enc_train), len(enc_val))
print("Encoded keys:", enc_train.column_names)  # should be ['input_ids','attention_mask','labels']
assert len(enc_train) > 0, "Encoded train is empty—check tokenization step."

# ---- Model + LoRA ----
base = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
lora_cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=8, lora_alpha=16, lora_dropout=0.05,
                      target_modules=["q","k","v","o"])
model = get_peft_model(base, lora_cfg)

dcoll = DataCollatorForSeq2Seq(tokenizer=tok, model=model)

args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-4,
    fp16=False,                 # keep False on CPU
    eval_strategy="epoch",      # if you're on 4.43 use evaluation_strategy="epoch"
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=50,
    report_to="none",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=enc_train,
    eval_dataset=enc_val,
    data_collator=dcoll,
    # processing_class=tok,     # optional (silences a future warning on v5)
    tokenizer=tok              # fine on current versions
)

trainer.train()

os.makedirs(OUT_DIR, exist_ok=True)
trainer.model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
print("Saved LoRA adapter to", OUT_DIR)
