import argparse, csv, json, os, random, re, html
from pathlib import Path
from bs4 import BeautifulSoup
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import json
import urllib.parse

# ---------- Utilities ----------

from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

def _fetch(url, headers):
    req = Request(url, headers=headers)
    return urlopen(req, timeout=20).read().decode("utf-8", errors="ignore")

def get_text(src: str) -> str:
    
    def is_short(s: str, n: int = 800) -> bool:
        return len((s or "").strip()) < n

    if src.startswith(("http://", "https://")):
        hdrs = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        def fetch(u): 
            return _fetch(u, hdrs)

        # ---------------- WIKIPEDIA ----------------
        if "wikipedia.org/wiki/" in src:
            slug = src.rsplit("/", 1)[-1]
            title = urllib.parse.unquote(slug).replace("_", " ")

            try:
                plain = fetch(f"https://en.wikipedia.org/api/rest_v1/page/plain/{slug}")
                plain = re.sub(r'\r\n?', '\n', plain)
                cleaned = clean_wiki_text(plain)
                if not is_short(cleaned, 1200):
                    return cleaned
            except Exception:
                pass

            try:
                mob = fetch(f"https://en.wikipedia.org/api/rest_v1/page/mobile-html/{slug}")
                text_only = re.sub(r"<[^>]+>", " ", mob)
                text_only = re.sub(r"\s+", " ", text_only)
                cleaned = clean_wiki_text(text_only)
                if not is_short(cleaned, 1200):
                    return cleaned
            except Exception:
                pass

            try:
                api = fetch("https://en.wikipedia.org/w/api.php"
                            f"?action=query&prop=extracts&explaintext=1&redirects=1&format=json&titles={urllib.parse.quote(title)}")
                obj = json.loads(api)
                pages = obj.get("query", {}).get("pages", {})
                extract = ""
                for _, page in pages.items():
                    if "extract" in page and page["extract"]:
                        extract = page["extract"]
                        break
                cleaned = clean_wiki_text(extract)
                if cleaned and not is_short(cleaned, 1200):
                    return cleaned
            except Exception:
                pass

            try:
                html_doc = fetch(src)
                soup = BeautifulSoup(html_doc, "html.parser")
                for el in soup.select("sup.reference, span.mw-editsection"):
                    el.decompose()
                root = soup.select_one("div.mw-parser-output") or soup
                paras = []
                for node in root.find_all(["p", "h2", "h3", "li"], recursive=True):
                    txt = node.get_text(" ", strip=True)
                    if not txt:
                        continue
                    if node.name in ("h2", "h3"):
                        paras.append(f"\n== {txt} ==")
                    else:
                        paras.append(txt)
                body = "\n".join(paras)
                cleaned = clean_wiki_text(body)
                if not is_short(cleaned, 1200):
                    return cleaned
            except Exception:
                pass

            try:
                data = json.loads(fetch(f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"))
                extract = (data.get("extract") or "").strip()
                cleaned = clean_wiki_text(extract)
                if cleaned:
                    return cleaned
            except Exception:
                pass

            return ""  #failed :(

        # --------------- NON-WIKIPEDIA ---------------
        try:
            html_doc = fetch(src)
        except Exception as e:
            raise FileNotFoundError(f"Could not fetch URL: {src} ({e})")

        soup = BeautifulSoup(html_doc, "html.parser")
        body = " ".join(x.get_text(" ", strip=True) for x in soup.find_all(["p","li","h2","h3"]))
        return re.sub(r"\s+", " ", body).strip()

    # --------------- LOCAL FILE ----------------
    p = Path(src)
    if p.exists():
        return p.read_text(encoding="utf-8", errors="ignore")
    raise FileNotFoundError(f"Source not found: {src}")

def clean_wiki_text(text: str) -> str:
    text = text.replace("\xa0", " ")

    text = re.sub(
        r'(?:^|\n)==\s*(References|See also|External links|Notes|Further reading)\s*==.*$',
        '',
        text,
        flags=re.I | re.S
    )

    text = re.sub(r'\s*\[(?:\d+|[a-z]|note\s*\d+|citation needed)\]\s*', ' ', text, flags=re.I)

    text = re.sub(r'\((?:citation needed|clarification needed|disputed|page needed|dead link|link rot)\)', ' ', text, flags=re.I)

    text = re.sub(r'\bISBN[:\s][0-9Xx\-–]+\b', ' ', text)
    text = re.sub(r'\bDOI:\s*[^\s,;]+', ' ', text, flags=re.I)
    text = re.sub(r'\bPMID\s*\d+\b', ' ', text, flags=re.I)
    text = re.sub(r'\bRetrieved\s+\d{1,2}\s+\w+\s+\d{4}\b', ' ', text, flags=re.I)

    text = re.sub(r'\bv\s*•\s*t\s*•\s*e\b', ' ', text, flags=re.I)

    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()

def is_content_sentence(s: str) -> bool:
    s = s.strip()

    # Must end like a real sentence
    if not re.search(r'[.!?]"?$', s):  # period/exclaim/question (optional closing quote)
        return False

    # Length & words
    if not (50 <= len(s) <= 350):
        return False
    if len(s.split()) < 8:
        return False

    if not re.search(r'\b(is|are|was|were|has|have|had|includes?|use[sd]?|named|called|described|developed|introduced|designed|built|created|founded|occurr?ed|took|won|published|works?|served|led|caused)\b', s, flags=re.I):
        return False

    bad_patterns = [
        r'\bISBN\b', r'\bDOI\b', r'\bPMID\b',
        r'\bRetrieved\b', r'Wayback Machine', r'Archive\.?org',
        r'www\.', r'https?://', r'\.com\b', r'\.org\b', r'\.edu\b',
        r'\bIn:\s', r'pcs\.c1\.Page', r'\bv\s*•\s*t\s*•\s*e\b',
        r'^[\"“][A-Z][^.!?]{0,100}[\"”]$',
        r'^[A-Z][A-Za-z0-9 ,–\-:]{0,120}$',
    ]
    if re.search("|".join(bad_patterns), s, flags=re.I):
        return False

    letters = re.findall(r'[A-Za-z]', s)
    if letters:
        upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
        if upper_ratio > 0.35:
            return False

    return True


def sent_split(text: str):
    chunks = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9(“"])', text)
    chunks = [c.strip() for c in chunks if c and len(c.strip()) >= 10]

    merged = []
    i = 0
    while i < len(chunks):
        s = chunks[i]
        if re.search(r'\b[A-Z]\.$', s) and i + 1 < len(chunks):
            s = (s + " " + chunks[i+1]).strip()
            i += 2
        else:
            i += 1
        merged.append(s)

    keep = [m for m in merged if is_content_sentence(m)]

    if len(keep) < 10:
        keep = [m for m in merged if 40 <= len(m) <= 350 and not re.search(r'(ISBN|DOI|PMID|www\.|https?://)', m, flags=re.I)]

    seen, out = set(), []
    for s in keep:
        k = s.lower()
        if k not in seen:
            seen.add(k); out.append(s)
    return out[:400]

def swap_entity(sentence: str, target: str, replacement: str) -> str:
    pattern = re.escape(target)
    return re.sub(pattern, replacement, sentence, count=1)

# ---------- Teacher components ----------

_fe = None
_ner = None
_t2t = None


def get_teacher_pipes():
    global _fe, _ner, _t2t
    if _fe is None:
        _fe = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
    if _ner is None:
        _ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    if _t2t is None:
        _t2t = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=96)
    return _fe, _ner, _t2t


def embed_sents(fe, sents):
    vecs = [np.array(fe(s, truncation=True)[0]).mean(axis=0) for s in sents]
    return np.stack(vecs)


def cosine(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def rank_sentences(sents, fe):
    M = embed_sents(fe, sents)
    centroid = M.mean(axis=0)
    scores = [cosine(v, centroid) for v in M]
    idx = np.argsort(scores)[::-1]
    return [sents[i] for i in idx]


def pick_entity(ner_res):
    pref = {"ORG":3,"PERSON":3,"LOC":3,"GPE":3,"EVENT":2,"DATE":2,"WORK_OF_ART":2,"LAW":2}
    best = None; bestw = -1
    for ent in ner_res:
        w = pref.get(ent.get("entity_group"), 1)
        word = ent.get("word", "").strip()
        if w > bestw and len(word) > 2:
            best, bestw = ent, w
    return best

def parse_distractors(raw: str, sentence: str):
    parts = re.split(r'[;\n]|(?:,\s(?=[A-Z]))', raw)
    alts, seen = [], set()

    for p in parts:
        t = p.strip(" -•\t\"'[]()")
        if not (2 <= len(t) <= 80):
            continue
        if re.search(r'(Alt\d|##|pcs\.c1|https?://|www\.)', t, flags=re.I):
            continue
        if not re.search(r'[A-Za-z]{3,}', t):
            continue
        if t in sentence:
            continue
        low = t.lower()
        if low in seen:
            continue
        seen.add(low)
        alts.append(t)

    return alts[:4]

def _norm_ent(s: str) -> str:
    s = s.replace("##", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _distinct3(a: str, b: str, c: str) -> bool:
    aa, bb, cc = a.strip(), b.strip(), c.strip()
    if len({aa, bb, cc}) < 3:
        return False
    return True

def _bad_text(s: str) -> bool:
    if not s or len(s) < 8:
        return True
    if re.search(r'(Alt\d|##|https?://|www\.)', s, re.I):
        return True
    letters = re.findall(r'[A-Za-z]', s)
    if letters and (sum(ch.isupper() for ch in letters) / len(letters)) > 0.45:
        return True
    return False

def display_clean(s: str) -> str:
    s = re.sub(r'\s*\[(?:\d+|[a-z]|note\s*\d+|citation needed|self-published source\?)\]\s*', ' ', s, flags=re.I)
    s = re.sub(r'\s*\. \.\s*', '. ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.replace('##', '')
    return s

def _distinct3(a: str, b: str, c: str) -> bool:
    a, b, c = a.strip(), b.strip(), c.strip()
    if not a or not b or not c:
        return False
    return len({a, b, c}) == 3

def _looks_like_title_or_link(s: str) -> bool:
    if re.search(r'(?:www\.|https?://|\.com\b|\.org\b|\.edu\b)', s, flags=re.I):
        return True
    if re.match(r'^[\"“][A-Z][^.!?]{0,60}[\"”]$', s):
        return True
    letters = re.findall(r'[A-Za-z]', s)
    if letters and (sum(1 for ch in letters if ch.isupper()) / len(letters)) > 0.45:
        return True
    return False


def list_entities(sentence: str, ner):
    pref = {"ORG":3,"PERSON":3,"LOC":3,"GPE":3,"EVENT":2,"DATE":2,"WORK_OF_ART":1,"LAW":1}
    ents = ner(sentence)
    seen = set(); out = []
    for e in ents:
        t = e.get("entity_group"); w = (e.get("word") or "").strip()
        if not t or len(w) < 3: 
            continue
        if w.lower() in seen: 
            continue
        seen.add(w.lower())
        out.append((w, t, pref.get(t, 0)))
    out.sort(key=lambda x: x[2], reverse=True)
    return [(w,t) for (w,t,_) in out]

def build_entity_pool(sents, ner):
    pool = {}
    for s in sents:
        for e in ner(s):
            g = e.get("entity_group")
            w = (e.get("word") or "").strip()
            if g and len(w) > 2:
                pool.setdefault(g, set()).add(w)
    return pool


def teacher_make_item(sentence: str, entity_pool=None, forced_entity=None):
    fe, ner, t2t = get_teacher_pipes()

    if forced_entity is not None:
        ent_text, ent_type = forced_entity
    else:
        ents = ner(sentence)
        anchor = pick_entity(ents)
        if not anchor:
            return None
        ent_text = anchor["word"]
        ent_type = anchor.get("entity_group")

    ent_text = _norm_ent(ent_text)

    prompt = (
        f"You are writing exam distractors. Given the sentence:\n"
        f"'{sentence}'\n"
        f"Propose TWO realistic but incorrect alternatives for the entity '{ent_text}'. "
        f"Each must be the same entity TYPE and plausibly confusable in context. "
        f"Return as a plain list separated by semicolons."
    )
    cand = t2t(prompt)[0]["generated_text"]
    alts = parse_distractors(cand, sentence)

    if len(alts) < 2:
        cand2 = t2t(prompt + "\nReturn exactly two alternatives in the format: Alt1; Alt2")[0]["generated_text"]
        alts = parse_distractors(cand2, sentence)

    if len(alts) < 2 and entity_pool and ent_type in entity_pool:
        candidates = [x for x in list(entity_pool[ent_type]) if _norm_ent(x).lower() != ent_text.lower()]
        random.shuffle(candidates)
        for c in candidates:
            c2 = _norm_ent(c)
            if c2 not in alts:
                alts.append(c2)
            if len(alts) >= 2:
                break

    alts = [a for a in alts if a][:2]
    if len(alts) < 2:
        return None

    wrong1 = swap_entity(sentence, ent_text, alts[0])
    wrong2 = swap_entity(sentence, ent_text, alts[1])

    if wrong1 == sentence or wrong2 == sentence:
        return None

    if (_bad_text(wrong1) or _bad_text(wrong2) or
        _looks_like_title_or_link(wrong1) or _looks_like_title_or_link(wrong2)):
        return None

    correct = sentence
    if not _distinct3(correct, wrong1, wrong2):
        return None

    correct = display_clean(correct)
    wrong1  = display_clean(wrong1)
    wrong2  = display_clean(wrong2)
    
    expl = t2t(
        f"Explain concisely (1–2 sentences each) why these are wrong in context:\n"
        f"Context: {sentence}\nWrong A: {wrong1}\nWrong B: {wrong2}"
    )[0]["generated_text"]

    return {
        "context": sentence,
        "entity": ent_text,
        "correct": correct,
        "distractors": [wrong1, wrong2],
        "explanations": [expl]
    }


# ---------- Student (adapter) inference ----------

_student_model = None
_student_tok = None


def load_student(adapter_dir: str):
    global _student_model, _student_tok
    if _student_model is None:
        base = "google/flan-t5-small"
        _student_tok = AutoTokenizer.from_pretrained(base)
        _student_model = AutoModelForSeq2SeqLM.from_pretrained(base)
        if adapter_dir and Path(adapter_dir).exists():
            from peft import PeftModel
            _student_model = PeftModel.from_pretrained(_student_model, adapter_dir)
        _student_model.eval()
    return _student_tok, _student_model


def student_generate(adapter_dir: str, sentence: str, entity_pool=None):
    tok, mdl = load_student(adapter_dir)
    prompt = (
        "Task: Given a factual sentence and a target entity from it, "
        "produce a JSON object with keys: correct (string), distractors (array of two strings), "
        "explanations (array of 1-2 short sentences). The distractors must be plausible but incorrect and of the same entity type.\n"
        f"Sentence: {sentence}\n"
        "Output strictly valid JSON."
    )
    ids = tok(prompt, return_tensors="pt")
    gen = mdl.generate(**ids, max_new_tokens=192)
    out = tok.decode(gen[0], skip_special_tokens=True).strip()
    try:
        obj = json.loads(out)
        if not isinstance(obj.get("correct"), str):
            raise ValueError
        if not isinstance(obj.get("distractors"), list) or len(obj["distractors"]) < 2:
            raise ValueError
        return {
            "context": sentence,
            "entity": None,
            "correct": obj["correct"],
            "distractors": obj["distractors"][:2],
            "explanations": obj.get("explanations", [])
        }
    except Exception:
        return teacher_make_item(sentence, entity_pool=entity_pool)

# ---------- Emitters ----------

def write_anki(cards, outdir: Path):
    csvp = outdir / "anki_flashcards.csv"
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Front","Back","Extra"])
        for c in cards:
            q = "Which statement is correct?"
            A = display_clean(c["correct"] or "")
            B = display_clean(c["distractors"][0] or "")
            C = display_clean(c["distractors"][1] or "")
            if not _distinct3(A, B, C):
                continue

            correct_slot = random.randint(0, 2)
            texts = [None, None, None]
            texts[correct_slot] = A
            rem = [i for i in (0,1,2) if i != correct_slot]
            texts[rem[0]] = B
            texts[rem[1]] = C

            opts = [(lab, texts[i], i == correct_slot) for i, lab in enumerate(["A","B","C"])]
            front = f"{q}<br><br>" + "<br>".join(f"{lab}) {html.escape(txt)}" for lab, txt, _ in opts)
            correct_letter = ["A","B","C"][correct_slot]
            exp = display_clean(' '.join(c.get('explanations', [])))
            back = f"Correct: {correct_letter}" + (f"<br><br>{html.escape(exp)}" if exp else "")
            extra = f"Context: {html.escape(display_clean(c['context']))}"
            w.writerow([front, back, extra])


def write_quiz(cards, outdir: Path):
    htmlp = outdir / "quiz.html"
    with open(htmlp, "w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'><title>WhyWrong Quiz</title>")
        f.write("<style>body{font-family:system-ui;margin:2rem;max-width:800px} .q{margin:1.5rem 0;padding:1rem;border:1px solid #ddd;border-radius:12px} .ans{display:none;margin:.5rem 0;color:#111;background:#f6f6f6;padding:.5rem;border-radius:8px}</style>")
        f.write("<h1>WhyWrong — Counterfactual Quiz</h1>")
        f.write("<p>Pick the correct statement. Click 'Check' to reveal the answer & explanation.</p>")
        f.write("<script>function chk(i,ans){let sel=[...document.querySelectorAll('input[name=q'+i+']:checked')][0];let box=document.getElementById('ans'+i);box.style.display='block';box.querySelector('b').innerText=(sel?sel.value:'—')===ans?'Correct':'Incorrect';}</script>")
        for i, c in enumerate(cards):
            q = "Which statement is correct?"
            A = display_clean(c["correct"] or "")
            B = display_clean(c["distractors"][0] or "")
            C = display_clean(c["distractors"][1] or "")
            if not _distinct3(A, B, C):
                continue

            correct_slot = random.randint(0, 2)
            texts = [None, None, None]
            texts[correct_slot] = A
            rem = [idx for idx in (0,1,2) if idx != correct_slot]
            texts[rem[0]] = B
            texts[rem[1]] = C

            correct_letter = ["A","B","C"][correct_slot]

            f.write(f"<div class='q'><h3>Q{i+1}. {html.escape(q)}</h3>")
            for lab, txt in zip(["A","B","C"], texts):
                f.write(f"<label><input type='radio' name='q{i}' value='{lab}'> {lab}) {html.escape(txt)}</label><br>")
            exp = display_clean(' '.join(c.get('explanations', [])))
            f.write(f"<button onclick=\"chk({i},'{correct_letter}')\">Check</button><p></p>")
            f.write(f"<div class='ans' id='ans{i}'><b></b><div><em>Explanation:</em> {html.escape(exp)}</div><div><em>Context:</em> {html.escape(display_clean(c['context']))}</div></div></div>")

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="URL or path to text")
    ap.add_argument("--k", type=int, default=10, help="number of cards to PRODUCE")
    ap.add_argument("--mode", choices=["teacher","student"], default="teacher")
    ap.add_argument("--adapters", default=None, help="path to LoRA adapters (for student mode)")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    text = get_text(args.source)
    sents = sent_split(text)
    print(f"[debug] sentences extracted = {len(sents)}; target cards k = {args.k}")

    cards = []

    fe, ner, _ = get_teacher_pipes()

    ranked = rank_sentences(sents, fe)

    pool = build_entity_pool(ranked[:120], ner)

    made = 0
    if args.mode == "teacher":
        for s in ranked:
            item = teacher_make_item(s, entity_pool=pool)
            if item:
                cards.append(item)
                made += 1
                if made >= args.k:
                    break
    else:
        for s in ranked:
            item = student_generate(args.adapters, s, entity_pool=pool)
            if item:
                cards.append(item)
                made += 1
                if made >= args.k:
                    break

    if not cards:
        print(f"[debug] sentences extracted = {len(sents)}; trying to make k={args.k}")
        raise SystemExit("No cards produced. Try a different source or increase --k.")

    outdir = Path(args.outdir)
    write_anki(cards, outdir)
    write_quiz(cards, outdir)
    (outdir/"report.json").write_text(json.dumps(cards, indent=2), encoding="utf-8")
    print(f"Wrote: {outdir/'anki_flashcards.csv'}, {outdir/'quiz.html'}, {outdir/'report.json'}")


if __name__ == "__main__":
    main()
