import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from syllables import count_syllables

def load_model(model_name: str):
    print(f"[load] model={model_name}")
    cfg = AutoConfig.from_pretrained(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)

    if getattr(cfg, "is_encoder_decoder", False):
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model_type = "seq2seq"
    else:
        mdl = AutoModelForCausalLM.from_pretrained(model_name)
        model_type = "causal"

    mdl.eval()
    print(f"[load] ready ({model_type})")
    return tok, mdl, model_type

# ---------- Prompt helpers ----------

_DEF_EXAMPLES = """\
Write a 3-line haiku in English with 5-7-5 syllables about the given topic.
Only output the poem (no extra text).

Example:
Topic: ocean
Haiku:
whales sing under stars
salt wind threads the drifting foam
tides carry old songs

Example:
Topic: spring
Haiku:
soft rain wakes the earth
buds lean into quiet light
sparrows stitch the dawn
"""

def _build_prompt(topic: str, line_mode: bool, target_syl: int | None = None):
    if line_mode:
        # Ask for a single line to fit a precise syllable count
        return (
            "Write one poetic line in English for a haiku.\n"
            f"Topic: {topic}\n"
            f"Target syllable count: {target_syl}\n"
            "Only output the line."
        )
    # Whole-haiku mode (we still post-filter syllables)
    return f"{_DEF_EXAMPLES}\nTopic: {topic}\nHaiku:"

# ---------- Generation core ----------

def _generate_text(prompt: str, tok, mdl, model_type: str, max_new=32, num_beams=8):
    ipt = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        if model_type == "seq2seq":
            out = mdl.generate(
                **ipt,
                max_new_tokens=max_new,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        else:
            out = mdl.generate(
                **ipt,
                max_new_tokens=max_new,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id
            )
    text = tok.decode(out[0], skip_special_tokens=True)
    return text.strip()

def _clean_line(s: str) -> str:
    s = s.strip()
    # keep simple punctuation; avoid side instructions
    for tag in ["Haiku:", "Line:", "Output:", "Answer:", "Response:" ]:
        if s.startswith(tag):
            s = s[len(tag):].strip()
    # take the first line only if multiple lines
    s = s.splitlines()[0].strip(" -–—.,;:!?'\"()[]{}")
    # normalize spaces
    return " ".join(s.split())

# ---------- Candidate search ----------

def _best_line(topic, target_syl, tok, mdl, model_type, tries=6):
    for i in range(tries):
        prompt = _build_prompt(topic, line_mode=True, target_syl=target_syl)
        raw = _generate_text(prompt, tok, mdl, model_type, max_new=20, num_beams=10)
        line = _clean_line(raw)
        if 0 < len(line) <= 80 and count_syllables(line) == target_syl:
            return line
    return None

def generate_haiku(topic: str, tok, mdl, model_type: str):
    # Try generating per-line with syllable targets (most reliable)
    l1 = _best_line(topic, 5, tok, mdl, model_type)
    l2 = _best_line(topic, 7, tok, mdl, model_type)
    l3 = _best_line(topic, 5, tok, mdl, model_type)

    if not (l1 and l2 and l3):
        # Fallback: generate full poem then clean per line & check
        prompt = _build_prompt(topic, line_mode=False)
        poem = _generate_text(prompt, tok, mdl, model_type, max_new=64, num_beams=8)
        lines = [ln for ln in (ln.strip() for ln in poem.splitlines()) if ln]
        if len(lines) >= 3:
            a, b, c = (_clean_line(lines[0]), _clean_line(lines[1]), _clean_line(lines[2]))
            if count_syllables(a) == 5 and count_syllables(b) == 7 and count_syllables(c) == 5:
                l1, l2, l3 = a, b, c

    # Final graceful fallbacks if any line is still missing
    l1 = l1 or f"{topic} in hush"
    l2 = l2 or "small echoes between leaves"
    l3 = l3 or "dawn learns to listen"
    return f"{l1}\n{l2}\n{l3}"
