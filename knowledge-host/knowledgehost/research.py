"""Parse Vinkona's research drops (research_loop_spec §2).

A solved doc is a single markdown file: YAML-ish front-matter + headed sections.
The host reads three things out of it:

  * the ``# Question`` — the frame distillation is conditioned on, so "how do I X"
    + source prose yields a *card answering X*, not a generic concept (§6.2);
  * the ``## Sources`` blocks — the untrusted evidence that actually gets chunked
    and distilled (the ``## Answer`` is Vinkona's synthesis, kept only as a fallback
    when a doc carries no sources);
  * front-matter ``kb_query`` — the verbatim query that opened the gap, so a card
    grounding it can close that ``knowledge_gap`` (§6.2).

Stdlib only (no PyYAML): we hand-parse the handful of scalar keys we need.
Everything here is treated as untrusted DATA — the host sanitises on ingest and
the filename is never interpolated anywhere.
"""
from __future__ import annotations

import re

_FM = re.compile(r"^﻿?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n", re.DOTALL)
_SCALAR = re.compile(r"^([A-Za-z0-9_]+):[ \t]*(.*?)[ \t]*$")


def _unquote(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Return (top-level scalar front-matter, body-after-front-matter).  Nested list
    keys (like ``sources:``) are skipped — the host doesn't need them."""
    m = _FM.match(text)
    if not m:
        return {}, text
    fm: dict = {}
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip() != line:   # skip nested/indented (list items)
            continue
        sm = _SCALAR.match(line)
        if sm and sm.group(2) != "":                     # "key:" with no scalar → a block/list
            fm[sm.group(1).lower()] = _unquote(sm.group(2))
    return fm, text[m.end():]


def is_research_doc(path: str) -> bool:
    """Cheap check: a markdown file whose front-matter declares provenance: vinkona."""
    if not path.lower().endswith((".md", ".markdown")):
        return False
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(2048)
    except OSError:
        return False
    fm, _ = parse_front_matter(head if "\n---" in head else head + "\n---\n")
    return (fm.get("provenance") or "").strip().lower() == "vinkona"


def _sections(body: str) -> list[tuple[int, str, str]]:
    """Split markdown into (level, heading, text) by ATX headings, in order."""
    out, level, head, buf = [], 0, "", []
    for line in body.splitlines():
        hm = re.match(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$", line)
        if hm:
            if head or buf:
                out.append((level, head, "\n".join(buf).strip()))
            level, head, buf = len(hm.group(1)), hm.group(2).strip(), []
        else:
            buf.append(line)
    if head or buf:
        out.append((level, head, "\n".join(buf).strip()))
    return out


def parse_research_doc(path: str) -> tuple[str, list, dict]:
    """Parse a solved doc → (question, blocks, meta).

    * ``question``  the ``# Question`` text (the distillation frame / chunk title).
    * ``blocks``    ``[(section, text)]`` for ``chunk_blocks`` — one per ``### source``
                    subsection under ``## Sources``; if the doc has no sources, a single
                    ``## Answer`` block so a card can still be distilled.
    * ``meta``      ``{provenance, kind, kb_query, question, answer, trust}``.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    fm, body = parse_front_matter(raw)
    secs = _sections(body)

    question = ""
    answer = ""
    source_blocks: list = []
    i = 0
    while i < len(secs):
        level, head, text = secs[i]
        h = head.lower()
        if level == 1 and h.startswith("question"):
            question = text.strip()
        elif h.startswith("answer"):
            answer = text.strip()
        elif h.startswith("source"):
            # its own prose (if any) plus every deeper subsection until the next
            # same-or-higher heading = the individual sources.
            if text.strip():
                source_blocks.append((head, text.strip()))
            j = i + 1
            while j < len(secs) and secs[j][0] > level:
                slvl, shead, stext = secs[j]
                if stext.strip():
                    source_blocks.append((f"{head} › {shead}", stext.strip()))
                j += 1
            i = j
            continue
        i += 1

    blocks = source_blocks
    if not blocks and answer:                            # no sources → distil the synthesis
        blocks = [("Answer", answer)]

    meta = {
        "provenance": (fm.get("provenance") or "vinkona").lower(),
        "kind": (fm.get("kind") or "").lower() or None,
        "kb_query": fm.get("kb_query") or None,
        "question": question or None,
        "answer": answer or None,
        "trust": fm.get("trust") or None,               # 'low' | number | None (label only)
    }
    return question, blocks, meta
