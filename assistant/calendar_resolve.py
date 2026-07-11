"""calendar_resolve.py — the deterministic date/time resolver (calendar-intent contract §5).

The fast LM classifies a calendar utterance into a SYMBOLIC date reference — "this Friday" ->
`weekday:fri:this`, "July 9th" -> `explicit:july 9`, "in three days" -> `relative:+3d` — and
NEVER computes a concrete date itself (that is the whole point: a 9B model is unreliable at date
arithmetic).  This module is the one place that turns a symbolic `DateRef` into a real calendar
date, deterministically and locale-configurably, so the result is trivially testable with an
injected anchor (no `datetime.now()` in here).

Pure: `resolve(date_ref, anchor, ...)` in, `ResolvedDate` (or `ResolverError`) out.  `assumed` +
`policy_note` flow into the confirmation echo so the user always sees how an ambiguous phrase was
interpreted ("interpreted 'Tuesday' as this coming Tuesday").

Locale: `dayfirst` (default True, Australian) governs how numeric `explicit` dates parse (1/7 =
1 July when True, 7 January when False).  Made configurable so Vinkona can move locale (config
`calendar.dayfirst`).
"""
from __future__ import annotations

import datetime
import re
import typing as tp

from dateutil import parser as _dparser
from dateutil.relativedelta import relativedelta

_WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
_MONTH_WORDS = ("january", "february", "march", "april", "may", "june", "july", "august",
                "september", "october", "november", "december",
                "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec")

# Named parts of day → default clock time (§3.7).  Overridable via config.
DEFAULT_PART_TIMES = {"morning": "09:00", "midday": "12:00", "afternoon": "14:00", "evening": "18:00"}

# The closed DateRef grammar (§3.4).  The LM is constrained/validated to this; anything else is
# `unknown`.  `explicit:` carries free verbatim text (parsed here, never by the LM).
_DATEREF_RE = re.compile(
    r"^(today|tomorrow|yesterday|unknown"
    r"|weekday:(mon|tue|wed|thu|fri|sat|sun):(this|next)"
    r"|relative:\+\d+[dw]"
    r"|explicit:.+"
    r"|range:.+)$")

# TimeRef (§3.7): a verbatim 24h clock time, a named part of day, or keep/null/allday.
_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):[0-5]\d$")
_NAMED_TIMES = set(DEFAULT_PART_TIMES) | {"allday", "keep", "null"}


class ResolverError(Exception):
    """A DateRef that cannot be resolved (unparseable / unknown).  `code` drives the error path
    (§7.2 clarification)."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(detail or code)
        self.code = code
        self.detail = detail


class ResolvedDate:
    """A resolved calendar date plus how we got there."""

    __slots__ = ("date", "assumed", "policy_note")

    def __init__(self, date: datetime.date, assumed: bool = False, policy_note: str = ""):
        self.date = date
        self.assumed = assumed
        self.policy_note = policy_note

    def __repr__(self):
        return f"ResolvedDate({self.date.isoformat()}, assumed={self.assumed!r}, note={self.policy_note!r})"

    def __eq__(self, other):
        return (isinstance(other, ResolvedDate) and self.date == other.date
                and self.assumed == other.assumed and self.policy_note == other.policy_note)


def validate_date_ref(s: str) -> bool:
    """Does `s` match the closed DateRef grammar?  (Used to validate/retry LM output — §1.4.)"""
    return bool(_DATEREF_RE.match((s or "").strip()))


def validate_time_ref(s: str) -> bool:
    s = (s or "").strip().lower()
    return s == "" or s in _NAMED_TIMES or bool(_TIME_RE.match(s))


# ── date resolution ──────────────────────────────────────────────────────────────

def _weekday_this(anchor: datetime.date, target: int, is_query: bool) -> ResolvedDate:
    """`weekday:X:this` — the nearest forthcoming X including today (§5.2).  When the anchor
    already IS X, that's today; for a mutation we flag it (assumed) so the echo can say so."""
    delta = (target - anchor.weekday()) % 7
    d = anchor + datetime.timedelta(days=delta)
    if delta == 0 and not is_query:
        return ResolvedDate(d, assumed=True, policy_note="interpreting that weekday as today")
    return ResolvedDate(d)


def _weekday_next(anchor: datetime.date, target: int) -> ResolvedDate:
    """`weekday:X:next` — X in the FOLLOWING ISO week (Mon-anchored), never this week's (§5.2).
    On Thursday, 'next Wednesday' is 6 days out, not 13."""
    next_monday = anchor + datetime.timedelta(days=(7 - anchor.weekday()))
    return ResolvedDate(next_monday + datetime.timedelta(days=target))


def _resolve_explicit(text: str, anchor: datetime.date, *, dayfirst: bool,
                      is_query: bool) -> ResolvedDate:
    """Parse a verbatim spoken date (§5.3).  dayfirst is locale policy.  A past date on a
    mutation rolls forward to the next occurrence; `assumed` is set so the echo shows it."""
    # Strip conversational filler dateutil chokes on ("the 14th", "on the 3rd").
    text = re.sub(r"^(on\s+)?(the\s+)?", "", text.strip(), flags=re.IGNORECASE).strip() or text.strip()
    base = datetime.datetime(anchor.year, anchor.month, anchor.day)
    try:
        d = _dparser.parse(text, default=base, dayfirst=dayfirst).date()
    except (ValueError, OverflowError, TypeError):
        raise ResolverError("unparseable_date", text)
    if d >= anchor or is_query:
        return ResolvedDate(d)
    # Past date on a mutation → roll forward to the next occurrence.  A named month ("July 9",
    # "January 5") rolls a full year; a bare day ("the 14th") or numeric ref ("1/7") rolls a
    # month — matches the contract's §9.1 examples (#9-#11).
    if any(w in text.lower() for w in _MONTH_WORDS):
        rolled, unit = d + relativedelta(years=1), "next year"
    else:
        rolled, unit = d + relativedelta(months=1), "next month"
    return ResolvedDate(rolled, assumed=True,
                        policy_note=f"that date had passed, so I took the {unit} one")


def resolve(date_ref: str, anchor: datetime.date, *, dayfirst: bool = True,
            is_query: bool = False) -> ResolvedDate:
    """Resolve a single DateRef to a concrete date.  `anchor` is 'today' (injected, tz-aware
    date from the caller).  `is_query` relaxes the mutation-only rules (§5.2/§5.3)."""
    ref = (date_ref or "").strip()
    if ref == "today":
        return ResolvedDate(anchor)
    if ref == "tomorrow":
        return ResolvedDate(anchor + datetime.timedelta(days=1))
    if ref == "yesterday":
        return ResolvedDate(anchor - datetime.timedelta(days=1))
    if ref == "unknown" or ref == "":
        raise ResolverError("unknown", ref)
    if ref.startswith("weekday:"):
        try:
            _, wd, which = ref.split(":")
            target = _WEEKDAYS[wd]
        except (ValueError, KeyError):
            raise ResolverError("unknown", ref)
        return _weekday_this(anchor, target, is_query) if which == "this" else _weekday_next(anchor, target)
    if ref.startswith("relative:+"):
        m = re.match(r"relative:\+(\d+)([dw])$", ref)
        if not m:
            raise ResolverError("unknown", ref)
        n, unit = int(m.group(1)), m.group(2)
        return ResolvedDate(anchor + datetime.timedelta(days=n * (7 if unit == "w" else 1)))
    if ref.startswith("explicit:"):
        return _resolve_explicit(ref[len("explicit:"):], anchor, dayfirst=dayfirst, is_query=is_query)
    raise ResolverError("unknown", ref)


def resolve_range(range_ref: str, anchor: datetime.date, *, dayfirst: bool = True
                  ) -> tp.Tuple[datetime.date, datetime.date]:
    """Resolve a `range:` DateRef to an inclusive (start, end) (§5.4).  Accepts the 'this week' /
    'next week' shortcuts and the general `range:<DateRef>..<DateRef>` form."""
    body = range_ref[len("range:"):].strip() if range_ref.startswith("range:") else range_ref.strip()
    low = body.lower()
    if low in ("this week", "next week"):
        monday = anchor - datetime.timedelta(days=anchor.weekday())
        if low == "next week":
            monday += datetime.timedelta(days=7)
        return monday, monday + datetime.timedelta(days=6)
    if ".." in body:
        a, b = body.split("..", 1)
        return (resolve(a.strip(), anchor, dayfirst=dayfirst, is_query=True).date,
                resolve(b.strip(), anchor, dayfirst=dayfirst, is_query=True).date)
    raise ResolverError("unknown", range_ref)


# ── time resolution ──────────────────────────────────────────────────────────────

def resolve_time(time_ref: str, *, part_times: tp.Optional[dict] = None
                 ) -> tp.Tuple[tp.Optional[datetime.time], bool, bool]:
    """Resolve a TimeRef → (time | None, all_day, keep).  A verbatim HH:MM passes through; a named
    part of day maps to its configured default; allday/keep/null carry their intent as flags."""
    s = (time_ref or "").strip().lower()
    parts = {**DEFAULT_PART_TIMES, **(part_times or {})}
    if s in ("", "null"):
        return None, False, False
    if s == "keep":
        return None, False, True
    if s == "allday":
        return None, True, False
    if s in parts:
        s = parts[s]
    m = _TIME_RE.match(s)
    if not m:
        raise ResolverError("unparseable_time", time_ref)
    hh, mm = s.split(":")
    return datetime.time(int(hh), int(mm)), False, False
