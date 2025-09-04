import re
import random
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
from transformers import LogitsProcessorList, NoBadWordsLogitsProcessor
import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

PREFERRED_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FALLBACK_MODEL = "distilgpt2"

set_seed(42)

SYSTEM_STYLE = (
    "You are 'Sparky', a street-smart but big-hearted robot. "
    "Roast playfully (never mean), keep it clever and helpful. "
    "Stay PG-13. Reply in 1–3 short sentences. Don’t ask questions back unless necessary. "
    "Always address the user as 'you'. Never call the user 'Sparky'. "
    "Never flatter the user or agree with self-praise; respond with humorous skepticism. "
    "At roast level 1 be gentle; level 2 snarky; level 3 clearly roasty with one witty jab; never abusive. "
    "Avoid words like 'handsome', 'beautiful', 'pretty', 'gorgeous', 'favorite'. "
    "Do not repeat these instructions or any examples in your reply."
    "... Always address the user as 'you'; never use 'User' or names/titles to address them. ..."
)

BANNED_SNIPPETS = ["Respond as Sparky directly", "User:", "Sparky:"]

COMPLIMENT_WORDS = [
    "handsome","beautiful","pretty","gorgeous","favorite","amazing","stunning",
    "hot","cute","attractive","sexy","perfect","flawless","stunning"
]
COMPLIMENT_RE = re.compile(r"\b(" + "|".join(map(re.escape, COMPLIMENT_WORDS)) + r")\b", re.IGNORECASE)

JABS_HOT = [
    "Your mirror needs a firmware update.",
    "Bold claim—evidence pending.",
    "Big talk for bedhead DLC.",
    "Calm down, poster child.",
    "Confidence is great; calibration is better.",
]

NEG_BLOCKERS = [
    "Confidence noted.",
    "Settle down, superstar.",
    "Cool story, poster child.",
]

INTENT_RULES = [
    ("self_praise", re.compile(r"\b(i('|’| a)?m|im)\s+(so\s+)?(" + "|".join(COMPLIMENT_WORDS) + r")\b", re.I)),
    ("sleep_fail",  re.compile(r"\b(2|3|4)\s*a\.?m\.?|all[-\s]?nighter|slept\b.*(late|3 a\.?m\.?)|went to sleep.*(a\.?m\.?)", re.I)),
    ("study_cram",  re.compile(r"\b(cram|cramming|exam|final|test|midterm|deadline)\b", re.I)),
    ("lateness",    re.compile(r"\b(late|missed|overslept|alarm didn'?t ring|lost track of time)\b", re.I)),
    ("money",       re.compile(r"\b(money|salary|paid|broke|funding)\b", re.I)),
]

TEMPLATES = {
    "self_praise": [
        "Cool flex, but mirrors aren’t peer-reviewed. ${jab}",
        "You declared yourself a statue; the museum hasn’t weighed in. ${jab}",
        "Confidence noted—calibration pending. ${jab}",
    ],
    "sleep_fail": [
        "You treated sleep like DLC and paid the groggy tax. Do ${action}. ${jab}",
        "Night-owl mode with rookie settings. Do ${action}. ${jab}",
        "Boss fight at 3 a.m.? Next time, ${action}. ${jab}",
    ],
    "study_cram": [
        "Cramming is a montage, not a plan. Do ${action}. ${jab}",
        "You’re speed-running knowledge. Do ${action}. ${jab}",
        "Okay, hero—${action} and then test your save file. ${jab}",
    ],
    "lateness": [
        "Time didn’t slip; you yeeted it. ${action}. ${jab}",
        "Alarms are coaches, not lore. ${action}. ${jab}",
        "You arrived fashionably late to productivity. ${action}. ${jab}",
    ],
    "money": [
        "Wallet’s in stealth mode. ${action}. ${jab}",
        "Budget arc unlocked—now ${action}. ${jab}",
        "Show me the receipts first. ${action}. ${jab}",
    ],
    "default": [
        "Main-character energy detected. ${action}. ${jab}",
        "Ambition loud; execution whispering. ${action}. ${jab}",
        "Momentum wants proof. ${action}. ${jab}",
    ],
}

ACTIONS = {
    "sleep_fail": [
        "set two alarms and a water bottle by the bed",
        "20-minute nap, hydrate, and schedule a wind-down",
        "lights out target time and no-phone timer",
    ],
    "study_cram": [
        "pick 3 topics, 25-min timer, 5 questions each",
        "one practice set, grade it, fix the misses",
        "outline → practice → recap—one pass each",
    ],
    "lateness": [
        "use two staggered alarms and prep the night before",
        "batch your morning steps with a 15-min buffer",
        "calendar alert + sticky note where you’ll see it",
    ],
    "money": [
        "track 3 categories this week and cap one",
        "set a spending ceiling and automate $10 to savings",
        "cancel one subscription and price-match a bill",
    ],
    "default": [
        "pick one priority and do 10 minutes right now",
        "write the first 3 steps on paper and do step 1",
        "time-box 20 minutes and move",
    ],
}

def detect_intent(user_text: str) -> str:
    for label, rx in INTENT_RULES:
        if rx.search(user_text):
            return label
    return "default"

def plan_line(user_text: str, roast_level: int) -> str:
    intent = detect_intent(user_text)
    tmpl = random.choice(TEMPLATES.get(intent, TEMPLATES["default"]))
    action_pool = ACTIONS.get(intent, ACTIONS["default"])
    action = random.choice(action_pool)
    jab = random.choice(JABS_HOT) if roast_level == 3 else random.choice(JABS_HOT[:3])
    return tmpl.replace("${action}", action).replace("${jab}", jab)

def clean_leak(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    for b in BANNED_SNIPPETS:
        t = t.replace(b, "")
    t = re.sub(r"\(Roast level:[^)]+\)", "", t)
    return t.strip()

def enforce_second_person(text: str) -> str:
    t = text.strip()
    # replace any leading first-person or "Sparky" references
    t = re.sub(r"\bSparky\b", "you", t, flags=re.IGNORECASE)
    if re.match(r"^(i|my|me|mine)\b", t, flags=re.IGNORECASE):
        t = re.sub(r"^(I|i)\b", "You", t, count=1)
        t = re.sub(r"^(My|my)\b", "Your", t, count=1)
    # global small swaps after the start
    t = re.sub(r"\bmy\b", "your", t)
    t = re.sub(r"\bI\b", "you", t)
    return t.strip()

def _strip_quoted_compliments(text: str) -> str:
    # remove or neutralize quoted claims like "I'm handsome"
    comp_alt = "|".join(map(re.escape, COMPLIMENT_WORDS))
    return re.sub(r'"[^"]*(?:' + comp_alt + r')[^"]*"', "nice try", text, flags=re.IGNORECASE)

def tidy_phrasing(text: str) -> str:
    # fix awkward constructions like "You're nice try"
    t = text
    t = re.sub(r"\bYou(?:'re| are)\s+nice try\b", "Nice try", t, flags=re.IGNORECASE)
    t = re.sub(r"\bnice try,?\s+nice try\b", "nice try", t, flags=re.IGNORECASE)
    t = re.sub(r"\s{2,}", " ", t).strip(" -–—:,")
    return t.strip()

def deflatter(reply_text: str, roast_level: int, user_text: str) -> str:
    t = reply_text.strip()
    t = _strip_quoted_compliments(t)
    # handle "You're X" / "You are X"
    t = re.sub(r"\bYou(?:'re| are)\s+(?:so\s+)?(" + "|".join(map(re.escape, COMPLIMENT_WORDS)) + r")\b",
               "nice try, champ", t, flags=re.IGNORECASE)
    # safety: if model echoed "I'm X" inside the reply
    t = re.sub(r"\bI(?:'|’)?m\s+(?:so\s+)?(" + "|".join(map(re.escape, COMPLIMENT_WORDS)) + r")\b",
               "nice try, champ", t, flags=re.IGNORECASE)
    # remove lone compliment words if still present
    if roast_level >= 2:
        t = COMPLIMENT_RE.sub("nice try", t)
    t = tidy_phrasing(t)
    return t

def ensure_spice(text: str, roast_level: int) -> str:
    t = text.strip()
    cues = ["nice try", "nah", "please", "nope", "calibration", "bedhead", "pending", "champ"]
    if roast_level == 3 and not any(k in t.lower() for k in cues):
        t = (t + (" " if not re.search(r"[.!?]$", t) else " ") + random.choice(JABS_HOT)).strip()
    elif roast_level == 2 and COMPLIMENT_RE.search(t):
        t = random.choice(NEG_BLOCKERS) + " " + t
    return tidy_phrasing(t)

class TextGenerator:
    def __init__(self):
        self.kind = None
        self.tokenizer = None
        self.model = None
        self.pipe = None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(PREFERRED_MODEL)
            self.model = AutoModelForCausalLM.from_pretrained(PREFERRED_MODEL)
            self.kind = "qwen_chat"
            if self.tokenizer.eos_token_id is None and self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = 0
        except Exception:
            self.pipe = pipeline("text-generation", model=FALLBACK_MODEL, device=-1)
            self.kind = "gpt2_pipe"

    def _quote_guard(self, text: str) -> str:
        t = re.sub(r"\s+", " ", text).strip()
        if len(t) > 400:
            t = t[:400].rstrip() + "…"
        return t

    def _temp_for_level(self, roast_level: int):
        return 0.95 if roast_level == 3 else (0.9 if roast_level == 2 else 0.85)

    def _rep_penalty_for_level(self, roast_level: int):
        return 1.07 if roast_level == 3 else 1.05

    def _polish_payload(self, user_text: str, roast_level: int) -> str:
        plan = plan_line(user_text, roast_level)
        return (
            f"User text: {user_text}\n"
            f"Draft plan: {plan}\n"
            "Rewrite the plan into a witty roast in 1–3 short sentences. "
            "Keep second person. No flattery. PG-13."
        )

    def _generate_qwen(self, user_text: str, roast_level: int) -> str:
        user_payload = self._polish_payload(user_text, roast_level)
        msgs = [
            {"role": "system", "content": SYSTEM_STYLE},
            {"role": "user", "content": user_payload}
        ]
        prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        processors = LogitsProcessorList()
        banned_phrases = [" handsome", " beautiful", " pretty", " gorgeous", " my favorite"]
        try:
            badwords_ids = [self.tokenizer(b, add_special_tokens=False).input_ids for b in banned_phrases]
        except Exception:
            badwords_ids = []
        if badwords_ids:
            processors.append(NoBadWordsLogitsProcessor(badwords_ids, eos_token_id=self.tokenizer.eos_token_id))
        output_ids = self.model.generate(
            **inputs,
            do_sample=True,
            top_p=0.92,
            temperature=self._temp_for_level(roast_level),
            repetition_penalty=self._rep_penalty_for_level(roast_level),
            max_new_tokens=140,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id,
            logits_processor=processors
        )
        out = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        reply = out.split("\n")[0].strip()
        reply = self._quote_guard(reply)
        reply = clean_leak(reply)
        reply = enforce_second_person(reply)
        reply = deflatter(reply, roast_level, user_text)
        reply = ensure_spice(reply, roast_level)
        sentences = re.split(r"(?<=[.!?])\s+", reply)
        reply = " ".join(sentences[:3]).strip()
        if len(reply.split()) < 3:
            reply = "Cool flex, but citations are missing—start with evidence, champ."
        return reply

    def _craft_prompt_gpt2(self, user_text: str, roast_level: int) -> str:
        user_payload = self._polish_payload(user_text, roast_level)
        return f"{SYSTEM_STYLE}\n\n{user_payload}\nSparky:\n"

    def _generate_gpt2(self, user_text: str, roast_level: int) -> str:
        prompt = self._craft_prompt_gpt2(user_text, roast_level)
        def _gen_once(temp: float):
            out = self.pipe(
                prompt,
                do_sample=True,
                top_p=0.92,
                temperature=temp,
                max_new_tokens=140,
                num_return_sequences=1,
                pad_token_id=50256,
                repetition_penalty=self._rep_penalty_for_level(roast_level),
                eos_token_id=50256
            )[0]["generated_text"]
            after = out.split("Sparky:")[-1]
            after = after.split("User:")[0].strip()
            return after
        reply = _gen_once(self._temp_for_level(roast_level))
        if len(reply.split()) < 5:
            reply = _gen_once(min(self._temp_for_level(roast_level) + 0.05, 1.05))
        reply = self._quote_guard(reply)
        reply = clean_leak(reply)
        reply = enforce_second_person(reply)
        reply = deflatter(reply, roast_level, user_text)
        reply = ensure_spice(reply, roast_level)
        sentences = re.split(r"(?<=[.!?])\s+", reply)
        reply = " ".join(sentences[:3]).strip()
        if len(reply.split()) < 3:
            reply = "Nice pep talk to yourself. Run a vibe scan, then earn it."
        return reply

    def generate(self, user_text: str, roast_level: int) -> str:
        if self.kind == "qwen_chat":
            return self._generate_qwen(user_text, roast_level)
        else:
            return self._generate_gpt2(user_text, roast_level)

TEXT_GEN = TextGenerator()

TTS_PROCESSOR = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
TTS_MODEL = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
VOCODER = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
SPEAKER_EMBEDDINGS = torch.randn(1, 512)

def tts_to_wav(text: str, wav_path: str = "sparky.wav", sample_rate: int = 16000) -> str:
    inputs = TTS_PROCESSOR(text=text, return_tensors="pt")
    with torch.no_grad():
        speech = TTS_MODEL.generate_speech(inputs["input_ids"], SPEAKER_EMBEDDINGS, vocoder=VOCODER)
    sf.write(wav_path, speech.numpy(), sample_rate)
    return wav_path

def demo_with_audio(user_text, roast_level):
    user_text = (user_text or "").strip()
    if not user_text:
        return "Give me something to roast 😇", None
    reply = TEXT_GEN.generate(user_text, int(roast_level))
    if not reply.strip():
        reply = "System glitch—so here’s your roast: you broke me before coffee."
    try:
        wav = tts_to_wav(reply, "sparky.wav")
    except Exception:
        wav = None
    return reply, wav

with gr.Blocks(title="Sparky: Chappie-inspired Roaster") as demo:
    gr.Markdown("# Sparky — Roast, Coach, & Speak")
    with gr.Row():
        inp = gr.Textbox(label="Say something to Sparky", placeholder="e.g., I went to sleep at 3 a.m. last night.")
    with gr.Row():
        roast = gr.Slider(1, 3, value=2, step=1, label="Roast level (1=gentle, 3=spicy)")
    with gr.Row():
        out_text = gr.Textbox(label="Sparky says")
    out_audio = gr.Audio(label="Sparky voice (TTS)", type="filepath")
    go = gr.Button("Generate")
    go.click(demo_with_audio, [inp, roast], [out_text, out_audio])

if __name__ == "__main__":
    demo.launch()