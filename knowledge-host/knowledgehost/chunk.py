"""Section-aware chunking + a stable, idempotent chunk id.

Sources hand us a list of ``(section_path, text)`` blocks (a heading and the
prose under it).  We pack each block into ~``chunk_target_tokens`` chunks, never
splitting across a heading unless a single section is too long — then we force a
split with a small overlap so a sentence straddling the cut is still findable in
both halves.

Token counts are estimated (chars/4), which is plenty for budgeting; the embed
model does the real tokenization.  The id is ``sha1(path + section + text)`` so
re-ingesting an unchanged document yields identical ids (idempotent upsert).
"""
from __future__ import annotations

import hashlib
import re

_SENT = re.compile(r"(?<=[.!?])\s+")


def est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def chunk_id(path_or_url: str, section: str, text: str) -> str:
    h = hashlib.sha1()
    h.update(path_or_url.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(section.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(text.encode("utf-8", "replace"))
    return h.hexdigest()[:20]


def _split_long(text: str, max_tokens: int, overlap_tokens: int):
    """Greedy sentence packing for one over-long section, with char overlap."""
    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4
    # Hard-cut any single "sentence" longer than the cap (tables / run-on text with
    # no .!? boundaries) so a chunk can't exceed the embed window via the no-split
    # path.  Leave room for the overlap tail the packer prepends on a flush.
    lim = max(1, max_chars - overlap_chars)
    sents = []
    for s in _SENT.split(text):
        while len(s) > lim:
            sents.append(s[:lim])
            s = s[lim:]
        sents.append(s)
    out, buf = [], ""
    for s in sents:
        if buf and len(buf) + len(s) + 1 > max_chars:
            out.append(buf.strip())
            tail = buf[-overlap_chars:] if overlap_chars else ""
            buf = (tail + " " + s).strip()
        else:
            buf = (buf + " " + s).strip() if buf else s
    if buf.strip():
        out.append(buf.strip())
    return out


def chunk_blocks(blocks, cfg: dict):
    """Yield dicts {section, text, tokens} ready for embedding/storage.

    `blocks` is an iterable of (section_path, text).  Adjacent short sections
    under the SAME heading are merged toward the target size; a section longer
    than chunk_max_tokens is sentence-split with overlap.
    """
    target = cfg["chunk_target_tokens"]
    hard = cfg["chunk_max_tokens"]
    overlap = cfg["chunk_overlap_tokens"]

    pending_section = None
    pending_text = ""

    def flush():
        nonlocal pending_text, pending_section
        text = pending_text.strip()
        pending_text = ""
        if not text:
            return
        if est_tokens(text) > hard:
            for piece in _split_long(text, hard, overlap):
                if piece.strip():
                    yield {"section": pending_section or "",
                           "text": piece, "tokens": est_tokens(piece)}
        else:
            yield {"section": pending_section or "",
                   "text": text, "tokens": est_tokens(text)}

    for section, text in blocks:
        text = (text or "").strip()
        if not text:
            continue
        section = section or ""
        # New heading, or the buffer is full -> flush what we have.
        same = (section == pending_section)
        if pending_text and (not same or est_tokens(pending_text) >= target):
            yield from flush()
        pending_section = section
        pending_text = (pending_text + "\n" + text).strip() if pending_text else text
        if est_tokens(pending_text) >= target:
            yield from flush()
    yield from flush()
