from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "microsoft/Phi-3-mini-4k-instruct"  # or google/flan-t5-small
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

def gen_copy(brief):
    prompt = f"""You are a concise marketing writer.
Business: {brief}
Return JSON with: headline(<=8w), tagline(<=12w), bullets(3 short), ctas(2).
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=220)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # quick json-safe extraction (or use a regex/json block)
    return parse_json(text)