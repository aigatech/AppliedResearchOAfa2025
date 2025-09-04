import re

_vowels = "aeiouy"
_dip = ("ia","io","eo","ua","ue","ui","uo","ii","ee","oo","ae","ai","ay","ey","oy","au","ou")

def _normalize(w):
    return re.sub(r"[^a-z]", "", w.lower())

def _count_word(w):
    w = _normalize(w)
    if not w:
        return 0
    if w in ("the",):
        return 1
    cnt = 0
    prev_v = False
    for ch in w:
        v = ch in _vowels
        if v and not prev_v:
            cnt += 1
        prev_v = v
    if w.endswith("e") and len(w) > 2 and not w.endswith(("le","ee","ye")):
        cnt = max(1, cnt-1)
    for d in _dip:
        cnt -= w.count(d)
    if w.endswith(("les","led")):
        cnt += 1
    return max(1, cnt)

def count_syllables(text):
    return sum(_count_word(t) for t in re.findall(r"[a-zA-Z']+", text))