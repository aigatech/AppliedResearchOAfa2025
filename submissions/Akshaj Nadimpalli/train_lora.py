import argparse, json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from inspect import signature

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", required=True)
    ap.add_argument("--output_dir", default="adapters/whywrong-lora")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--max_source_len", type=int, default=768)
    ap.add_argument("--max_target_len", type=int, default=256)
    args = ap.parse_args()

    base = "google/flan-t5-small"
    tok = AutoTokenizer.from_pretrained(base)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(base)

    ds = load_dataset("json", data_files=args.train_path, split="train")
    ds = ds.train_test_split(test_size=0.1, seed=42)

    def tok_fn(batch):
        model_in = tok(batch["input"], truncation=True, max_length=args.max_source_len)
        labels = tok(text_target=batch["target"], truncation=True, max_length=args.max_target_len)
        model_in["labels"] = labels["input_ids"]
        return model_in

    ds_tok = ds.map(tok_fn, batched=True, remove_columns=ds["train"].column_names)

    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q","v","k","o"]  # common for T5
    )
    mdl = get_peft_model(mdl, peft_cfg)

    collator = DataCollatorForSeq2Seq(tok, model=mdl)

    kw = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=50,
        remove_unused_columns=True,
        report_to=[],
    )
    
    sig = signature(TrainingArguments)
    if "evaluation_strategy" in sig.parameters:
        kw["evaluation_strategy"] = "epoch"
    if "save_strategy" in sig.parameters:
        kw["save_strategy"] = "epoch"
    if "fp16" in sig.parameters:
        kw["fp16"] = False
    if "bf16" in sig.parameters:
        kw["bf16"] = False
    
    train_args = TrainingArguments(**kw)

    trainer = Trainer(
        model=mdl,
        args=train_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["test"],
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()
    mdl.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapters to {args.output_dir}")

if __name__ == "__main__":
    main()
