# summarizer.py

from __future__ import annotations
import os, re, warnings
from typing import List, Dict, Optional
from transformers import AutoTokenizer, pipeline
from transformers.utils.logging import set_verbosity_error as hf_set_verbosity_error

warnings.filterwarnings("ignore")               
hf_set_verbosity_error()                        
try:
    from tokenizers import logging as tk_logging
    tk_logging.set_verbosity_error()            
except Exception:
    pass

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")       # force CPU
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")  # quiet HF hub

_T5 = "google/flan-t5-small"

_SECTION_ALIASES = {
    "Abstract": ["abstract", "summary"],
    "Introduction": ["introduction", "background"],
    "Methods": ["method", "methods", "approach", "model"],
    "Results": ["results", "experiments", "evaluation", "findings"],
    "Discussion": ["discussion", "analysis", "interpretation"],
    "Limitations": ["limitations", "limits", "failure cases"],
    "Conclusion": ["conclusion", "future work", "outlook"],
}

class Summarizer:
    def __init__(self, chunk_tokens: int = 420, chunk_stride: int = 120):
        self.tok = AutoTokenizer.from_pretrained(_T5)
        self.gen = pipeline("text2text-generation", model=_T5, device=-1) 
        self.chunk_tokens = chunk_tokens
        self.chunk_stride = chunk_stride

    def _token_chunks(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        ids = self.tok.encode(text, add_special_tokens=False)
        chunks, n, i, step = [], len(ids), 0, max(1, self.chunk_tokens - self.chunk_stride)
        while i < n:
            window = ids[i : i + self.chunk_tokens]
            chunks.append(self.tok.decode(window, skip_special_tokens=True))
            if i + self.chunk_tokens >= n:
                break
            i += step
        return chunks

    def _gen(self, prompt: str, max_new_tokens: int) -> str:
        return self.gen(
            prompt,
            do_sample=False,
            num_beams=4,
            max_new_tokens=max_new_tokens,
            clean_up_tokenization_spaces=True,
        )[0]["generated_text"].strip()

    # (1) Abstract Summary
    def summarize(self, text: str) -> Dict[str, str]:
        chunks = self._token_chunks(text)
        if not chunks:
            return {"num_chunks": "0", "summary": ""}
        prompt = (
            "Summarize concisely for a ~90-second spoken script. "
            "Use a hook, 2–3 key beats, and a takeaway.\n\n"
            f"{chunks[0]}"
        )
        out = self._gen(prompt, max_new_tokens=180)
        return {"num_chunks": str(len(chunks)), "summary": out}

    # (2) One concise question per requested section (keyword match with aliases)
    def _find_section_chunks(self, text: str, sections: Optional[List[str]] = None):
        if sections is None:
            sections = list(_SECTION_ALIASES.keys())
        chunks = self._token_chunks(text)
        if not chunks:
            return []
        lowers = [c.lower() for c in chunks]
        found = []
        for section in sections:
            aliases = _SECTION_ALIASES.get(section, [section.lower()])
            best_idx, best_score = None, -1
            for i, lc in enumerate(lowers):
                score = sum(1 for a in aliases if a in lc)
                if score > best_score:
                    best_score, best_idx = score, i
            if best_idx is not None and best_score > 0:
                found.append((section, best_idx, chunks[best_idx]))
        return found

    def questions_for_sections(
        self, text: str, sections: Optional[List[str]] = None, max_new_tokens: int = 48
    ) -> List[Dict[str, object]]:
        picked = self._find_section_chunks(text, sections)
        results: List[Dict[str, object]] = []
        for section, idx, chunk in picked:
            prompt = (
                f"Write ONE concise, insightful question that a student might ask "
                f"after reading the {section} section below. The question must be "
                f"answerable from the text and avoid yes/no.\n\n"
                f"{chunk}"
            )
            q = self._gen(prompt, max_new_tokens=max_new_tokens)
            q = re.split(r"[\r\n]+", q.strip())[0]
            if not q.endswith("?"):
                q = q.rstrip(".") + "?"
            results.append({"section": section, "chunk_index": idx, "question": q})
        return results
