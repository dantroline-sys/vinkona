"""HTML — prefer ``trafilatura`` (strips site chrome); fall back to a stdlib
``html.parser`` that keeps text and splits on ``<h1>``..``<h3>`` headings.

Shared by ``epub`` (each chapter is HTML) and reused conceptually by the
Wikipedia ZIM path (whose articles are pre-rendered HTML).
"""
from __future__ import annotations

import os
import re
from html.parser import HTMLParser


def _trafilatura(raw: str):
    try:
        import trafilatura
    except Exception:
        return None
    txt = trafilatura.extract(raw, include_comments=False, include_tables=True)
    return txt


class _Sectioner(HTMLParser):
    """Collect text into ``(heading_path, text)`` blocks, breaking on h1-h3."""
    _SKIP = {"script", "style", "head", "nav", "footer", "header", "aside"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks = []
        self.title = ""
        self._stack = []
        self._cur = ""
        self._buf = []
        self._skip_depth = 0
        self._in_h = 0
        self._h_level = 0
        self._h_text = []
        self._in_title = False

    def _flush(self):
        text = re.sub(r"[ \t]+", " ", " ".join(self._buf)).strip()
        if text:
            self.blocks.append((self._cur, text))
        self._buf = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in ("h1", "h2", "h3"):
            self._flush()
            self._in_h = 1
            self._h_level = int(tag[1])
            self._h_text = []

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in ("h1", "h2", "h3") and self._in_h:
            heading = " ".join(self._h_text).strip()
            self._stack[:] = self._stack[:self._h_level - 1] + [heading]
            self._cur = " > ".join([h for h in self._stack if h])
            if self._h_level == 1 and not self.title:
                self.title = heading
            self._in_h = 0

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
        elif self._in_h:
            self._h_text.append(data)
        else:
            self._buf.append(data)

    def result(self):
        self._flush()
        return self.title.strip(), [(s, t) for s, t in self.blocks if t.strip()]


def extract_from_string(raw: str, fallback_title: str = ""):
    title, blocks = "", None
    body = _trafilatura(raw)
    if body:
        # trafilatura returns clean text; keep it as one (untitled) block stream
        blocks = [("", body)]
    if blocks is None:
        p = _Sectioner()
        try:
            p.feed(raw)
            title, blocks = p.result()
        except Exception:
            blocks = [("", re.sub(r"<[^>]+>", " ", raw))]
    return (title or fallback_title), (blocks or [("", "")])


def extract(path: str, cfg: dict):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    fallback = os.path.splitext(os.path.basename(path))[0]
    return extract_from_string(raw, fallback)
