"""spoken_time.py — render dates/times as words a TTS engine will read naturally.

Vinkona's calendar confirmations are built deterministically (llm_bridge `_run_write`), so the
machine format the host hands back — "Sat 04 Jul 07:00–13:30" / "2026-07-04T07:00" — would go
straight to the voice as "zero seven colon zero zero".  This turns a start (and optional end) into
spoken English:

    2026-07-04T07:00 .. 13:30  ->  "Saturday the fourth of July, from seven in the morning to one
                                    thirty in the afternoon"
    2026-07-04T07:00           ->  "Saturday the fourth of July, at seven o'clock in the morning"
    2026-07-04 (date only)     ->  "Saturday the fourth of July"

Stdlib-only and pure, so it unit-tests without the model stack.
"""
from __future__ import annotations

import datetime
import typing as tp

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty"}


def _num_word(n: int) -> str:
    """0..59 in words: 5 -> five, 21 -> twenty-one."""
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    w = _TENS[tens]
    return w if ones == 0 else f"{w}-{_ONES[ones]}"


def _ordinal_word(n: int) -> str:
    """1..31 as a spoken ordinal: 4 -> fourth, 21 -> twenty-first, 30 -> thirtieth."""
    special = {1: "first", 2: "second", 3: "third", 5: "fifth", 8: "eighth",
               9: "ninth", 12: "twelfth"}
    if n in special:
        return special[n]
    if n < 20:
        return _ONES[n] + "th"
    tens, ones = divmod(n, 10)
    if ones == 0:
        return _TENS[tens][:-1] + "ieth"          # twenty -> twentieth, thirty -> thirtieth
    return f"{_TENS[tens]}-{_ordinal_word(ones)}"  # twenty-first, thirty-first


def _part_of_day(hour: int) -> str:
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    if hour < 21:
        return "evening"
    return "night"


def _clock_word(dt: datetime.datetime, *, sharp_oclock: bool = False) -> str:
    """A wall-clock time in words (12-hour, no am/pm — part-of-day carries that): 07:00 -> seven
    (or 'seven o'clock' when sharp_oclock), 13:30 -> one thirty, 09:05 -> nine oh five."""
    h12 = dt.hour % 12 or 12
    m = dt.minute
    if m == 0:
        return f"{_num_word(h12)} o'clock" if sharp_oclock else _num_word(h12)
    if m < 10:
        return f"{_num_word(h12)} oh {_num_word(m)}"
    return f"{_num_word(h12)} {_num_word(m)}"


def parse(s: str) -> tp.Tuple[tp.Optional[datetime.datetime], bool]:
    """Parse an ISO-ish datetime/date string → (datetime|None, has_time).  Fail-soft: an
    unparseable string returns (None, False) so the caller can fall back to raw prose."""
    s = (s or "").strip()
    if not s:
        return None, False
    has_time = ("T" in s) or (":" in s)
    core = s.replace("Z", "").strip()
    try:
        return datetime.datetime.fromisoformat(core), has_time
    except Exception:
        try:
            return datetime.datetime.strptime(core[:10], "%Y-%m-%d"), False
        except Exception:
            return None, False


def _spoken_date(dt: datetime.datetime) -> str:
    return f"{dt:%A} the {_ordinal_word(dt.day)} of {dt:%B}"


def spoken_range(start: datetime.datetime, has_time: bool,
                 end: tp.Optional[datetime.datetime] = None) -> str:
    """Compose the full spoken phrase for a (start[, end]) appointment."""
    date_part = _spoken_date(start)
    if not has_time:                                  # all-day / date-only
        return date_part
    same_time = end is not None and (end.hour, end.minute) == (start.hour, start.minute)
    if end is None or same_time:
        return f"{date_part}, at {_clock_word(start, sharp_oclock=True)} in the {_part_of_day(start.hour)}"
    p1, p2 = _part_of_day(start.hour), _part_of_day(end.hour)
    if p1 == p2:
        return f"{date_part}, from {_clock_word(start)} to {_clock_word(end)} in the {p1}"
    return (f"{date_part}, from {_clock_word(start)} in the {p1} "
            f"to {_clock_word(end)} in the {p2}")


def spoken_when(start_s: str, end_s: str = "") -> tp.Optional[str]:
    """Convenience: parse strings and render, or None if the start won't parse (caller keeps its
    own raw fallback then)."""
    start, has_time = parse(start_s)
    if start is None:
        return None
    end, _ = parse(end_s)
    return spoken_range(start, has_time, end)
