"""Per-option help for the Settings UI, extracted from config.py itself.

config.py's DEFAULTS block already documents every option — each key carries
the comment a maintainer would write into a help system, and it stays honest
because it lives next to the code that reads the option.  Rather than
duplicating that prose into a second file that would drift, the UI's
per-field help is EXTRACTED from those comments at request time and served
keyed by dotted config path ("tts.orpheus_gguf.repeat_penalty" → text).

To edit a setting's help: edit its comment in config.py.  That's the whole
authoring workflow — the extractor picks up leading comment lines, the
trailing same-line comment, and far-right continuation lines below it.

An OVERLAY file (help.json, same folder) supplies everything that isn't a
config key — tab intros keyed "tab.<name>" — and can override any extracted
path when different wording is wanted in the UI.  Both sources are re-read
when their file changes, so editing help never needs a restart.
"""
from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from pathlib import Path

_DECOR = re.compile(r"[─═┈]{2,}")          # box-drawing runs in section-banner comments


def _clean(parts: list) -> str:
    txt = " ".join(p for p in parts if p)
    txt = _DECOR.sub(" ", txt)
    return re.sub(r"\s{2,}", " ", txt).strip()


def _comments(src: str) -> dict:
    """{row: (col, text)} for every comment token (max one per line)."""
    out = {}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            out[tok.start[0]] = (tok.start[1], tok.string.lstrip("#").strip())
    return out


def _full_line(lines: list, row: int, col: int) -> bool:
    return lines[row - 1][:col].strip() == ""


def _walk(node: ast.Dict, path: list, lines: list, comments: dict, out: dict) -> None:
    for k, v in zip(node.keys, node.values):
        if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
            continue
        dp = ".".join(path + [k.value])
        krow, kcol = k.lineno, k.col_offset
        parts: list = []

        # Leading full-line comments at the key's own indent.  A comment far to
        # the RIGHT of the indent is the continuation of the PREVIOUS key's
        # trailing comment, not this key's doc — the col check excludes it.
        lead: list = []
        r = krow - 1
        while r in comments:
            col, text = comments[r]
            if not _full_line(lines, r, col) or col > kcol + 2:
                break
            lead.append(text)
            r -= 1
        parts += reversed(lead)

        # Trailing comments on the key/value lines themselves.  A dict value
        # spans its children's rows, so only its own key line counts there.
        vend = krow if isinstance(v, ast.Dict) else (v.end_lineno or krow)
        for row in range(krow, vend + 1):
            if row in comments and (row == krow or not _full_line(lines, row, comments[row][0])
                                    or comments[row][0] > kcol + 2):
                parts.append(comments[row][1])

        # Far-right full-line continuations below a scalar value.
        if not isinstance(v, ast.Dict):
            r = vend + 1
            while r in comments:
                col, text = comments[r]
                if not _full_line(lines, r, col) or col <= kcol + 2:
                    break
                parts.append(text)
                r += 1

        txt = _clean(parts)
        if txt:
            out[dp] = txt
        if isinstance(v, ast.Dict):
            _walk(v, path + [k.value], lines, comments, out)


def extract(config_py) -> dict:
    """DEFAULTS comments from `config_py` → {dotted.path: help text}."""
    src = Path(config_py).read_text()
    tree = ast.parse(src)
    target = None
    for n in tree.body:
        if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "DEFAULTS":
            target = n.value
        elif isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "DEFAULTS" for t in n.targets):
            target = n.value
    if not isinstance(target, ast.Dict):
        return {}
    out: dict = {}
    _walk(target, [], src.splitlines(), _comments(src), out)
    return out


# mtime-keyed caches so /api/help stays cheap but edits show on refresh
_cache: dict = {}


def _cached(path: Path, loader):
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    key = str(path)
    hit = _cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        data = loader(path)
    except Exception:
        data = {}
    _cache[key] = (mtime, data)
    return data


def load(base_dir) -> dict:
    """The merged help map: extracted config.py comments ⊕ help.json overlay
    (overlay wins).  Re-reads either file when it changes."""
    base = Path(base_dir)
    merged = dict(_cached(base / "config.py", extract))
    merged.update(_cached(base / "help.json", lambda p: json.loads(p.read_text())))
    return merged
