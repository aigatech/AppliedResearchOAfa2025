# scripts/download_model.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
MODEL_NAME = "google/flan-t5-small"
OUT_DIR = "./models/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained(OUT_DIR)
model.save_pretrained(OUT_DIR)

print("Saved model and tokenizer to", OUT_DIR)
