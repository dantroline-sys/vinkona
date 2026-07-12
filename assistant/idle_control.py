"""
Idle-work suppression: manual pause/resume + scheduled quiet hours.

Vinkona's research worker does background work on the fast and big LMs when the
box is idle.  Sometimes you want those LMs free — e.g. so the knowledge host can
distill uninterrupted.  This module decides, from a manual override plus a list
of scheduled quiet windows, whether idle work should be SUPPRESSED right now.

Two inputs:
  • override — a manual switch the header button sets: "" (follow the schedule),
    "paused" (force off), or "active" (force on, ignore the schedule).
  • quiet_hours — a list of {"start": "HH:MM", "end": "HH:MM"} windows during
    which idle work is suppressed when the override is "auto".  A window may wrap
    midnight (start > end), e.g. 22:00–06:00.

Pure and dependency-free so it's easy to unit-test; the worker and the config
server both use it so the UI and the actual behaviour never disagree.
"""

from __future__ import annotations


def parse_hm(s: str) -> int | None:
    """"HH:MM" -> minutes since midnight (0–1439), or None if unparseable."""
    try:
        h, m = str(s).strip().split(":")
        h, m = int(h), int(m)
    except Exception:
        return None
    if not (0 <= h < 24 and 0 <= m < 60):
        return None
    return h * 60 + m


def in_window(now_min: int, start: str, end: str) -> bool:
    """Is now_min inside [start, end)?  Handles a window that wraps midnight
    (start > end, e.g. 22:00–06:00).  start == end means an empty window (never),
    not all-day — a full-day quiet block is expressed as 00:00–24:00 by using
    end '00:00' with start '00:00' is ambiguous, so callers wanting all-day use
    two windows or simply the manual 'paused' override."""
    s, e = parse_hm(start), parse_hm(end)
    if s is None or e is None or s == e:
        return False
    if s < e:
        return s <= now_min < e
    # wraps midnight: inside if after start OR before end
    return now_min >= s or now_min < e


def in_any_window(now_min: int, windows: list) -> bool:
    for w in windows or []:
        if in_window(now_min, w.get("start", ""), w.get("end", "")):
            return True
    return False


def now_minutes(lt) -> int:
    """Minutes since local midnight from a time.struct_time (time.localtime())."""
    return lt.tm_hour * 60 + lt.tm_min


def is_suppressed(override: str, now_min: int, quiet_hours: list) -> bool:
    """The single source of truth: should idle work be suppressed right now?"""
    ov = (override or "").strip().lower()
    if ov == "paused":
        return True
    if ov == "active":
        return False
    return in_any_window(now_min, quiet_hours)   # "auto" / unset → follow schedule


def describe(override: str, now_min: int, quiet_hours: list) -> dict:
    """A small status dict for the UI: effective state + why + the next flip time
    (minutes-of-day) so the header can show 'paused until 14:00'."""
    ov = (override or "auto").strip().lower() or "auto"
    if ov not in ("auto", "paused", "active"):
        ov = "auto"
    suppressed = is_suppressed(ov, now_min, quiet_hours)
    reason = ("manually paused" if ov == "paused" else
              "forced active" if ov == "active" else
              "quiet hours" if suppressed else "idle work active")
    nxt = _next_boundary(now_min, quiet_hours) if ov == "auto" else None
    return {"override": ov, "suppressed": suppressed, "reason": reason,
            "next_change_min": nxt}


def _next_boundary(now_min: int, quiet_hours: list) -> int | None:
    """The nearest upcoming window edge (start or end) after now, as minutes-of-day.
    Used only for display ('active until 10:00' / 'paused until 14:00')."""
    edges = []
    for w in quiet_hours or []:
        for key in ("start", "end"):
            m = parse_hm(w.get(key, ""))
            if m is not None:
                edges.append(m)
    if not edges:
        return None
    ahead = sorted((e - now_min) % 1440 for e in edges if (e - now_min) % 1440 != 0)
    return (now_min + ahead[0]) % 1440 if ahead else None
