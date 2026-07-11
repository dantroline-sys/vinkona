"""Plain text and Markdown — stdlib only.

Markdown is split into sections on ATX headings (``#``..``######``), carrying a
heading path; plain ``.txt`` becomes one untitled block (chunk.py packs it).
"""
from __future__ import annotations

import os
import re

_H = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def extract(path: str, cfg: dict):
    raw = _read(path)
    title = os.path.splitext(os.path.basename(path))[0]
    if os.path.splitext(path)[1].lower() != ".md":
        return title, [("", raw)]

    blocks, stack, buf = [], [], []
    cur_section = ""

    def flush():
        if buf:
            blocks.append((cur_section, "\n".join(buf).strip()))
            buf.clear()

    for line in raw.splitlines():
        m = _H.match(line)
        if m:
            flush()
            level = len(m.group(1))
            heading = m.group(2).strip()
            stack[:] = stack[:level - 1] + [heading]
            cur_section = " > ".join(stack)
            if level == 1 and not blocks:
                title = heading or title
        else:
            buf.append(line)
    flush()
    blocks = [(s, t) for s, t in blocks if t.strip()]
    return title, blocks or [("", raw)]
