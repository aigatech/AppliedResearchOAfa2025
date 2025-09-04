import os, re, sys, math, argparse, warnings
from typing import List, Dict
from collections import Counter
from transformers import pipeline
from transformers.utils import logging as hf_logging

# Quiet logs
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Models (CPU-friendly)
EMOTION_MODEL = "joeddav/distilbert-base-uncased-go-emotions-student"
SUMM_MODEL    = "sshleifer/distilbart-cnn-12-6"
REPLY_MODEL   = "google/flan-t5-small"
NER_MODEL     = "dslim/bert-base-NER"

STYLES = ["coach", "genz", "pirate", "formal"]

STOP = {
    "the","a","an","and","or","but","so","to","of","in","on","for","at","by","with",
    "is","am","are","was","were","be","been","being","i","you","he","she","it","we","they",
    "this","that","these","those","as","from","if","then","than","too","very","just","not",
    "do","does","did","have","has","had","will","would","can","could","should","may","might",
    "about","into","over","under","again","also","there","here","up","down","yo","bro","dude",
}

GENZ_TOKENS   = ["fr", "vibes", "low-key", "bet", "ong", "✨", "🔥"]
PIRATE_TOKENS = ["Arrr", "matey", "aye", "ahoy", "Avast"]


def read_input(s: str) -> str:
    if os.path.isfile(s):
        with open(s, "r", encoding="utf-8") as f:
            return f.read().strip()
    return s.strip()

def ascii_bar(score: float, width: int = 20) -> str:
    blocks = int(round(score * width))
    return "█" * blocks + "-" * (width - blocks)

def is_question(text: str) -> bool:
    t = text.strip().lower()
    return t.endswith("?") or t.startswith((
        "how","what","why","when","where","can ","could ","will ","would ","should "
    ))

def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z\-']+", text.lower())

def top_keywords(text: str, k: int = 6) -> List[str]:
    toks = [w for w in tokenize_words(text) if w not in STOP and len(w) >= 3]
    if not toks: return []
    return [w for w, _ in Counter(toks).most_common(k)]

def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def split_sents(t: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]

def anti_echo_filter(reply: str, user: str, thresh: float = 0.55) -> str:
    user_sents = split_sents(user.lower())
    user_toks  = [tokenize_words(s) for s in user_sents]
    kept = []
    for s in split_sents(reply):
        toks = tokenize_words(s.lower())
        if not toks:
            continue
        sim = max((jaccard(toks, ut) for ut in user_toks), default=0.0)
        if sim < thresh:
            kept.append(s)
    return " ".join(kept).strip()

# ---------- pipelines ----------
def load_pipelines():
    emo = pipeline("text-classification", model=EMOTION_MODEL, return_all_scores=True)
    summ = pipeline("summarization", model=SUMM_MODEL)
    gen  = pipeline("text2text-generation", model=REPLY_MODEL)
    ner  = pipeline("ner", model=NER_MODEL, aggregation_strategy="simple")
    return emo, summ, gen, ner

def top_emotions(emo_output: List[List[Dict]], k: int = 5) -> List[Dict]:
    scores = emo_output[0]; scores.sort(key=lambda d: d["score"], reverse=True)
    return scores[:k]

def tldr(summarizer, text: str) -> str:
    words = text.split()
    if len(words) < 20:
        return text
    src = text if len(text) < 1100 else text[:1100]
    out = summarizer(src, max_length=40, min_length=12, do_sample=False)[0]["summary_text"]
    out = re.sub(r"\s+([?.!,])", r"\1", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def style_header(style: str) -> str:
    if style == "genz":
        return ("Reply in one short Gen-Z sentence (<=18 words). Be specific. "
                "Avoid quoting or repeating the user's wording. Include at least one of: fr, vibes, low-key, bet, ong, ✨, 🔥.")
    if style == "pirate":
        return ("Reply in one short pirate sentence (<=22 words). Be specific. "
                "Avoid quoting or repeating the user's wording. Include at least one of: Arrr, matey, aye, ahoy, Avast.")
    if style == "formal":
        return ("Write 1–2 concise professional sentences. Be specific. "
                "Do not quote or repeat the user's wording. No slang; no exclamation marks.")
    # coach handled separately (deterministic)
    return ""

def build_prompt(style: str, context: str, question: bool, anchors: List[str]) -> str:
    header = style_header(style)
    ask = "The user asked a question—answer briefly." if question else "The user made a statement—respond helpfully."
    hint = f"If natural, mention one of: {', '.join(anchors)}." if anchors else "Reference the topic explicitly."
    return f"{header}\n{ask}\n{hint}\n\nCONTEXT:\n{context}\n\nASSISTANT:"

def enforce_style_tokens(style: str, text: str) -> str:
    t = text.strip()
    if style == "genz" and not any(tok.lower() in t.lower() for tok in GENZ_TOKENS):
        t = t.rstrip(".") + " fr ✨"
    if style == "pirate" and not any(tok.lower() in t.lower() for tok in [x.lower() for x in PIRATE_TOKENS]):
        t = "Arrr, " + t.lstrip()
    if style == "formal":
        t = t.replace("!", ".")
    return t

def looks_like_empty_or_echo(user: str, reply: str) -> bool:
    if len(reply.split()) < 3: 
        return True
    sim = jaccard(tokenize_words(user.lower()), tokenize_words(reply.lower()))
    return sim >= 0.7

# ---------- COACH ----------
TIME_WORDS = ["tomorrow", "today", "tonight", "now", "this week", "next week"]

def detect_time_phrase(text: str) -> str:
    tl = text.lower()
    for w in TIME_WORDS:
        if w in tl:
            return w
    return ""

EXCLUDE_TOPIC = {
    "help","please","thanks","thank","badly","really","very","lot","always","true","friend","friends"
}

def choose_topic(kws: List[str]) -> str:
    # special combos
    s = set(kws)
    if "english" in s and "test" in s: return "your English test"
    if "math" in s and "test" in s:    return "your math test"
    if "project" in s:                 return "the project"
    if "assignment" in s or "essay" in s: return "the assignment"
    # pick first content word
    for w in kws:
        if w not in EXCLUDE_TOPIC:
            article = "your " if w in {"resume","interview","portfolio"} else "the "
            return (article + w)
    return "this"

def coach_reply(user_text: str, kws: List[str], question: bool) -> str:
    when = detect_time_phrase(user_text)
    topic = choose_topic(kws)
    when_str = f" {when}" if when else ""
    if question:
        return f"Yes—we'll work on {topic}{when_str}. First, we'll review mistakes and drill the weak spots."
    # statement
    return f"We'll tackle {topic}{when_str}. Next, we'll list the issues and practice short reps."


def main():
    ap = argparse.ArgumentParser(description="VibeGauge — Personalized Emo Bars + Styled Replies (coach fixed)")
    ap.add_argument("message", help="Your message, or a path to a .txt file")
    ap.add_argument("--style", choices=STYLES, default="coach", help="Reply style")
    args = ap.parse_args()

    user_text = read_input(args.message)
    if not user_text:
        print("No input text found."); sys.exit(0)

    emo, summ, gen, ner = load_pipelines()

    # Emotions
    emo_raw = emo(user_text); emo_top = top_emotions(emo_raw, k=5)

    # Context (for non-coach styles)
    context = tldr(summ, user_text)

    # Anchors (soft, for non-coach styles)
    ents = [e["word"].strip() for e in ner(user_text) if (e.get("entity_group") or "").upper() in {"PER","ORG","LOC","MISC"}]
    seen=set(); ents=[x for x in ents if not (x in seen or seen.add(x))]
    kws  = top_keywords(user_text, k=6)
    anchors = (ents + kws)[:6]
    question = is_question(user_text)

    # ----- COACH again cz it wasnt workin so lowk gota add a failsafe lol -------
    if args.style == "coach":
        reply = coach_reply(user_text, kws, question)

    else:
        # ----- other styles: generate with anti-echo settings -----
        prompt = build_prompt(args.style, context, question, anchors)
        out = gen(
            prompt,
            max_new_tokens=60,
            num_beams=6,
            do_sample=False,
            length_penalty=0.9,
            repetition_penalty=1.35,
            no_repeat_ngram_size=3,
            encoder_no_repeat_ngram_size=4,
            early_stopping=True,
            eos_token_id=gen.tokenizer.eos_token_id,
        )[0]["generated_text"].strip()

        cleaned = anti_echo_filter(out, user_text, thresh=0.55)
        reply = cleaned if cleaned else out
        if looks_like_empty_or_echo(user_text, reply):
            reply = {
                "genz":   "say less—big respect, you’re heard fr ✨",
                "pirate": "Arrr, ye be true crew, matey—stand fast together.",
                "formal": "Understood. Your message is appreciated; I’ll respond constructively.",
            }[args.style]
        reply = enforce_style_tokens(args.style, reply)

    # ----- Output -----
    print("\n=== INPUT ===")
    print(user_text)

    print("\n=== EMOTIONS ===")
    for item in emo_top:
        label = item["label"].capitalize(); score = float(item["score"])
        print(f"{label:12s} {score:0.2f}  {ascii_bar(score)}")

    print("\n=== TL;DR ===")
    print(context)

    print(f"\n=== REPLY ({args.style}) ===")
    print(reply)

if __name__ == "__main__":
    main()
