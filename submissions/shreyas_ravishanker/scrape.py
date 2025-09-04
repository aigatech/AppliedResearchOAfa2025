#!/usr/bin/env python3

import io
import os
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader


UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


class Scraper:
    def __init__(self, source, timeout=20, user_agent=UA):
        self.source = source
        self.timeout = timeout
        self.user_agent = user_agent

    def extract(self):
        if self._is_url(self.source):
            data, ctype = self._fetch_url_bytes(self.source)
            if "pdf" in ctype or self._is_pdf_bytes(data) or self.source.lower().endswith(".pdf"):
                return self._pdf_bytes_to_text(data)
            return self._html_to_text(self._decode_html(data))
        if not os.path.exists(self.source):
            raise FileNotFoundError(f"File not found: {self.source}")
        data, ctype = self._read_local(self.source)
        if "pdf" in ctype:
            return self._pdf_bytes_to_text(data)
        return self._html_to_text(self._decode_html(data))

    @staticmethod
    def _is_url(s):
        try:
            p = urlparse(s)
            return p.scheme in ("http", "https") and bool(p.netloc)
        except Exception:
            return False

    def _fetch_url_bytes(self, url):
        r = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=self.timeout)
        r.raise_for_status()
        return r.content, (r.headers.get("Content-Type") or "").lower()

    @staticmethod
    def _is_pdf_bytes(b):
        return b.startswith(b"%PDF")

    @staticmethod
    def _decode_html(b):
        try:
            return b.decode("utf-8", errors="ignore")
        except Exception:
            return b.decode("latin-1", errors="ignore")

    @staticmethod
    def _html_to_text(html):
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style", "noscript", "template", "svg", "canvas",
                       "iframe", "header", "footer", "nav", "aside", "form"]):
            t.decompose()
        for br in soup.find_all("br"):
            br.replace_with("\n")
        text = soup.get_text(separator="\n")
        lines = [ln.rstrip() for ln in text.splitlines()]
        out, blanks = [], 0
        for ln in lines:
            if ln.strip():
                blanks = 0
                out.append(ln)
            else:
                blanks += 1
                if blanks <= 2:
                    out.append("")
        return "\n".join(out).strip()

    @staticmethod
    def _pdf_bytes_to_text(b):
        reader = PdfReader(io.BytesIO(b))
        pages = []
        for p in reader.pages:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            if t:
                pages.append(t)
        text = "\n\n".join(pages).strip()
        return "\n".join(line.rstrip() for line in text.splitlines())

    @staticmethod
    def _read_local(path):
        with open(path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf" or Scraper._is_pdf_bytes(data):
            return data, "application/pdf"
        return data, "text/html"
