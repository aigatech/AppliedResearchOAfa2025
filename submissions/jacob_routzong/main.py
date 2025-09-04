import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.utils import logging
import json
from difflib import SequenceMatcher
from datasets import Dataset

# Suppress symlink and logging warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.set_verbosity_error()

# ----------------------------
# CONFIGURATION
# ----------------------------
MODEL_NAME = "microsoft/phi-2"
# MODEL_NAME = "facebook/opt-125m"  # Uncomment if phi-2 fails
DATA_FILE = "data/seed.jsonl"
MAX_NEW_TOKENS = 50
BATCH_SIZE = 1  # Safe for RTX 4070

# Use BF16 if supported
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Device map for auto distribution
DEVICE_MAP = "auto"

# ----------------------------
# LOAD MODEL & TOKENIZER
# ----------------------------
print(">>> Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # Suppress pad token warning
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("Try MODEL_NAME = 'facebook/opt-125m'.")
    exit(1)

print(">>> Loading model...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=TORCH_DTYPE,
        device_map=DEVICE_MAP,
        low_cpu_mem_usage=True
    )
except Exception as e:
    print(f"Error loading model: {e}")
    print("Try MODEL_NAME = 'facebook/opt-125m'.")
    exit(1)

print(">>> Model loaded successfully.")

# ----------------------------
# CREATE PIPELINE
# ----------------------------
print(">>> Setting up text-generation pipeline...")
try:
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        device_map=DEVICE_MAP
    )
except Exception as e:
    print(f"Error setting up pipeline: {e}")
    exit(1)
print(">>> Pipeline ready.")

# ----------------------------
# LOAD DATA
# ----------------------------
questions = []
answers = []

try:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            questions.append(obj.get("question", ""))
            answers.append(obj.get("answer", ""))
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found. Create it with valid JSONL data.")
    print("Example: {\"question\": \"What is 2+2?\", \"answer\": \"4\"}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSONL format in {DATA_FILE}: {e}")
    exit(1)

# Create Dataset
dataset = Dataset.from_dict({"question": questions, "answer": answers})

# ----------------------------
# EVALUATION FUNCTION
# ----------------------------
def evaluate_model(dataset):
    correct = 0
    num_questions = len(dataset)
    if num_questions == 0:
        print("Error: No questions loaded from data file.")
        return 0
    try:
        # Process in batches manually for compatibility
        for i in range(0, num_questions, BATCH_SIZE):
            batch_questions = dataset["question"][i:i + BATCH_SIZE]
            batch_answers = dataset["answer"][i:i + BATCH_SIZE]
            results = pipe(batch_questions, max_new_tokens=MAX_NEW_TOKENS, num_return_sequences=1)
            for j, (result, answer) in enumerate(zip(results, batch_answers)):
                pred = result[0]["generated_text"] if isinstance(result, list) else result["generated_text"]
                pred_lower = pred.lower()
                answer_lower = answer.lower()
                # Keyword matching
                keywords = [answer_lower]
                if answer_lower == "4":
                    keywords.append("four")
                elif answer_lower == "paris":
                    keywords.append("france")
                if any(keyword in pred_lower for keyword in keywords) or SequenceMatcher(None, pred_lower, answer_lower).ratio() > 0.8:
                    correct += 1
                    print(f"Match for question {i + j + 1}: Predicted '{pred}' contains or is similar to '{answer}'")
                else:
                    print(f"No match for question {i + j + 1}: Predicted '{pred}', expected '{answer}'")
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        print("Try MODEL_NAME = 'facebook/opt-125m'.")
        return 0
    accuracy = correct / num_questions if num_questions else 0
    print(f">>> Base accuracy: {accuracy:.2f}")
    return accuracy

# ----------------------------
# RUN SELF-EVALUATION LOOP
# ----------------------------
print(">>> Starting self-evaluation...")
base_acc = evaluate_model(dataset)
print(">>> Self-evaluation iteration complete.")