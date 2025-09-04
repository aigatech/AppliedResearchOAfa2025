# prompt_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch, sys, re
from random import choice



MODEL_DIR = "yo-mama-gpt/final"  # ← change if your save path is different

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

# GPT-2 has no pad token by default; use eos as pad
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# pipeline picks CUDA if available; CPU otherwise
device = 0 if torch.cuda.is_available() else -1
gen = pipeline("text-generation", model=model, tokenizer=tok, device=device)

def one_liner(prompt, max_new=16):
    out = gen(
        prompt,
        max_new_tokens=max_new,
        do_sample=True,          # flip to False for deterministic
        temperature=0.6,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.3,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        num_return_sequences=1,
    )[0]["generated_text"]
    # keep only first sentence after the prompt
    completion = re.split(r'(?<=[.!?])\s', out[len(prompt):].strip())[0]
    return prompt + completion

if __name__ == "__main__":
    print("""
    _______________________
    Welcome to YO-MAMA-GPT!
    Click Enter, and I'll 
    generate a random joke.
    Have fun!
    _______________________
    """)
    random = input("Press Enter to generate a random joke: ")
    YO_MAMA_PROMPTS = [
        "Yo mama so clumsy,",
        "Yo mama so smart,",
        "Yo mama so forgetful,",
        "Yo mama so fast,",
        "Yo mama so slow,",
        "Yo mama so strong,",
        "Yo mama so weak,",
        "Yo mama so tall,",
        "Yo mama so short,",
        "Yo mama so old,",
        "Yo mama so fat,",
        "Yo mama so broke,",
        "Yo mama so rich,",
        "Yo mama so brave,",
        "Yo mama so scared,",
        "Yo mama so loud,",
        "Yo mama so quiet,",
        "Yo mama so messy,",
        "Yo mama so tidy,",
        "Yo mama so sleepy,",
        "Yo mama so caffeinated,",
        "Yo mama so hungry,",
        "Yo mama so picky,",
        "Yo mama so generous,",
        "Yo mama so stubborn,",
        "Yo mama so curious,",
        "Yo mama so dramatic,",
        "Yo mama so chill,",
        "Yo mama so extra,",
        "Yo mama so basic,",
        "Yo mama so organized,",
        "Yo mama so chaotic,",
        "Yo mama so overprepared,",
        "Yo mama so underprepared,",
        "Yo mama so creative,",
        "Yo mama so distracted,",
        "Yo mama so punctual,",
        "Yo mama so late,",
        "Yo mama so thrifty,",
        "Yo mama so spendy,",
        "Yo mama so competitive,",
        "Yo mama so supportive,",
        "Yo mama so spicy,",
        "Yo mama so sweet,",
        "Yo mama so salty,",
        "Yo mama so cheesy,",
        "Yo mama so corny,",
        "Yo mama so cool,",
        "Yo mama so nerdy,",
        "Yo mama so athletic,",
    ]
    print(one_liner(choice(YO_MAMA_PROMPTS)))
