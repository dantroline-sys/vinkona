"""Shared "what is Vinkona doing" state + graceful-preemption helpers.

Two processes coordinate through cheap, atomic files/rows — no RPC:

  • the CASCADE (live chat) writes activity.json {open, kind, ts} on connect / each
    turn / close.  The idle worker treats a live session as 'not idle' and stands
    down; the config panel reads it to show "In a chat".
  • the idle WORKER writes a worker_state 'activity' row {doing, label, since,
    interruptible} naming its current task, so every UI can say what she's doing
    ("Reading the news", "Distilling chat history", "Standing down to talk…").

The coordination decisions live here as PURE functions so they're unit-testable
without a live box (the sandbox can't run the cascade/worker).  Callers own the
I/O (file read/write, the worker_state row); this module only shapes and reads.

Time is passed in (`now`) rather than read here, so tests are deterministic.
"""

import json

# ── activity.json : cascade → worker + UIs ────────────────────────────────────


def session_record(open_: bool, kind, now: float) -> dict:
    """The activity.json payload the cascade writes.  `kind` is 'text'/'audio'/None."""
    return {"open": bool(open_), "kind": kind, "ts": float(now)}


def read_session(text, now: float, open_stale: float) -> dict:
    """Parse activity.json → a normalized view for status/display.

    Returns {active, kind, open, stale, age_s}.  A session left 'open' but silent
    for open_stale seconds is treated as abandoned (active False, stale True) — the
    same rule the idle worker uses so the two never disagree.  Missing/torn text ⇒
    no session."""
    base = {"active": False, "kind": None, "open": False, "stale": False, "age_s": None}
    if not text:
        return base
    try:
        d = json.loads(text)
    except (ValueError, TypeError):
        return {**base, "unreadable": True}
    age = max(0.0, float(now) - float(d.get("ts", 0)))
    is_open = bool(d.get("open"))
    stale = age >= float(open_stale)
    return {"active": is_open and not stale, "kind": d.get("kind"),
            "open": is_open, "stale": stale, "age_s": age}


def should_yield(text, now: float, open_stale: float) -> bool:
    """True when a LIVE session is open right now, so idle big-LM work should stand
    down at its next checkpoint.  Fail-toward-yield on a torn/unreadable file: never
    fight a possibly-live session for the shared GPU.  A missing file ⇒ nothing is
    running ⇒ don't yield."""
    if not text:
        return False
    try:
        d = json.loads(text)
    except (ValueError, TypeError):
        return True
    if not d.get("open"):
        return False
    age = max(0.0, float(now) - float(d.get("ts", 0)))
    return age < float(open_stale)


# ── worker_state 'activity' : worker → UIs ────────────────────────────────────

# Stable keys for the worker's current task, with a plain-language label the UIs
# show verbatim.  The worker names its step from this so wording stays consistent
# across the panel, the phone app, and supervisor status.
LABELS = {
    "idle": "Idle — waiting quietly",
    "reflect": "Reflecting on what she's learned",
    "mind_graph": "Distilling chat history into long-term memory",
    "consolidate": "Tidying and merging what she knows",
    "research": "Researching a question",
    "crawl": "Reading your mail and files in the background",
    "ingest": "Taking in your connected tools",
    "rss": "Checking the news",
    "news_digest": "Writing the daily news digest",
    "export": "Syncing research to the knowledge library",
    "garden": "Gardening her knowledge base",
    "reconcile": "Reorganising memory",
    "calendar_sync": "Tidying your calendar",
    "orient": "Catching up on the world (weather, news, calendar)",
    "standing_down": "Standing down to talk with you…",
}


def label_for(doing: str, detail: str = "") -> str:
    """Human label for a task key, optionally suffixed with a detail (e.g. the
    research topic).  Unknown keys fall back to the key itself so it's never blank."""
    base = LABELS.get(doing, doing.replace("_", " ").capitalize() if doing else "Working")
    detail = (detail or "").strip()
    return f"{base}: {detail}" if detail else base


def activity_record(doing: str, now: float, *, detail: str = "",
                    interruptible: bool = True) -> dict:
    """The worker_state 'activity' payload the worker writes at each task."""
    return {"doing": doing, "label": label_for(doing, detail),
            "interruptible": bool(interruptible), "since": float(now)}


def read_activity(text, now: float):
    """Parse the worker_state 'activity' row → dict with an added age_s, or None."""
    if not text:
        return None
    try:
        d = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(d, dict):
        return None
    d = dict(d)
    d["age_s"] = max(0.0, float(now) - float(d.get("since", 0)))
    return d
