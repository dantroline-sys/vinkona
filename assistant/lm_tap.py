"""
lm_tap.py — a live feed of what the language models are seeing and saying.

Every LM call (fast or big, any process — cascade, worker, panel) can drop its context and
its output here, and the config panel's Live tab streams it.  Built for flying-blind
debugging: when the toolsmith says a build failed, or a reply comes out strange, the exact
prompt that went in and the exact text that came out are one click away.

Storage is a RAM file, never the disk: /dev/shm on Linux (a tmpfs — bytes live in memory,
vanish on reboot), the OS temp dir elsewhere.  It's a size-capped JSONL ring: writers
append one line per event and trim the head when the cap is passed, so it can run forever.
Multiple processes append concurrently (one O_APPEND write per event; the reader skips a
rare torn line rather than erroring).

An event: {ts, id, src: big|fast, kind: request|response|error, lane, model, text,
elapsed_s?, meta?}.  `text` is clipped head+tail so a giant prompt can't bloat the feed.

Stdlib only.  Fail-soft everywhere: a feed problem must never cost an LM call.
"""
from __future__ import annotations

import json
import os
import tempfile
import time

# "All of it, so it can be inspected": the caps exist only to stop a pathological event
# from eating the ring — real prompts and outputs fit whole.  It's RAM (tmpfs), so a few
# MB is nothing.
EVENT_HEAD = 200_000     # per-event text: keep this much of the head…
EVENT_TAIL = 50_000      # …and this much of the tail (prompts end with the question)
MAX_BYTES = 12_000_000   # trim the feed once it passes this…
TRIM_TO = 8_000_000      # …keeping roughly this much tail

LIVE_STALE_S = 300       # a live slot older than this is a crashed writer — ignore it

PATH_OVERRIDE: str | None = None       # tests point this at a temp file


def feed_path() -> str:
    if PATH_OVERRIDE:
        return PATH_OVERRIDE
    base = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    try:
        uid = os.getuid()
    except AttributeError:                          # Windows
        uid = 0
    return os.path.join(base, f"vinkona-lm-feed-{uid}.jsonl")


def _clip(text: str) -> str:
    text = str(text or "")
    if len(text) <= EVENT_HEAD + EVENT_TAIL + 64:
        return text
    dropped = len(text) - EVENT_HEAD - EVENT_TAIL
    return (text[:EVENT_HEAD] + f"\n… [{dropped} chars clipped] …\n" + text[-EVENT_TAIL:])


def write(src: str, kind: str, text: str, *, call_id: str = "", lane: str = "",
          model: str = "", elapsed_s: float | None = None, meta: dict | None = None) -> None:
    """Append one event.  Best-effort by contract — never raises."""
    try:
        ev = {"ts": round(time.time(), 3), "id": call_id, "src": src, "kind": kind,
              "lane": lane, "model": model, "text": _clip(text)}
        if elapsed_s is not None:
            ev["elapsed_s"] = round(float(elapsed_s), 2)
        if meta:
            ev["meta"] = meta
        p = feed_path()
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        if os.path.getsize(p) > MAX_BYTES:
            _trim(p)
    except Exception:
        pass


def _trim(path: str) -> None:
    try:
        with open(path, "rb") as f:
            f.seek(-TRIM_TO, os.SEEK_END)
            tail = f.read()
        nl = tail.find(b"\n")                       # drop the torn first line
        tail = tail[nl + 1:] if nl >= 0 else tail
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(tail)
        os.replace(tmp, path)                       # writers reopen per event → pick up the new file
    except Exception:
        pass


def read(n: int = 120, src: str | None = None) -> list:
    """The most recent events, oldest→newest, optionally filtered to one source.  Reads
    only the file tail, tolerates torn/garbage lines."""
    try:
        p = feed_path()
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            if size > TRIM_TO:
                f.seek(-TRIM_TO, os.SEEK_END)
                f.readline()                        # skip the torn line
            data = f.read()
    except OSError:
        return []
    out = []
    for line in data.decode("utf-8", "replace").splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict) and (src is None or ev.get("src") == src):
            out.append(ev)
    return out[-max(1, int(n)):]


def find_event(call_id: str, kind: str) -> dict | None:
    """One full stored event by id+kind — the panel's 'show all' fetch."""
    for ev in reversed(read(n=1000)):
        if ev.get("id") == call_id and ev.get("kind") == kind:
            return ev
    return None


# ── the live slot: the call that is streaming RIGHT NOW ──────────────────────
# One slot per source, overwritten in place as tokens arrive, so the panel can show output
# WHILE the model writes it (a 3-minute code generation is watchable, not a black box).
# The final response event supersedes it; a crashed writer's slot goes stale and is ignored.

def live_path(src: str) -> str:
    p = feed_path()
    return os.path.join(os.path.dirname(p),
                        os.path.basename(p).replace(".jsonl", "") + f"-live-{src}.json")


def stream(src: str, call_id: str, text: str, *, lane: str = "", model: str = "") -> None:
    """Overwrite the live slot with the output accumulated so far.  Callers throttle;
    best-effort by contract — never raises."""
    try:
        p = live_path(src)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps({"ts": round(time.time(), 3), "id": call_id,
                                "lane": lane, "model": model, "text": _clip(text)},
                               ensure_ascii=False))
        os.replace(tmp, p)                          # readers never see a torn write
    except Exception:
        pass


def live_read(src: str) -> dict | None:
    try:
        with open(live_path(src), encoding="utf-8") as f:
            ev = json.load(f)
        if not (isinstance(ev, dict) and ev.get("text")):
            return None
        if time.time() - float(ev.get("ts") or 0) > LIVE_STALE_S:
            return None                             # a crashed writer left it behind
        return ev
    except Exception:
        return None


def live_clear(src: str) -> None:
    try:
        os.remove(live_path(src))
    except OSError:
        pass


def clear() -> None:
    try:
        os.remove(feed_path())
    except OSError:
        pass
    for src in ("big", "fast"):
        live_clear(src)
