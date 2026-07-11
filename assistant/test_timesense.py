#!/usr/bin/env python
"""
Tests for timesense.py — the semantic clock (time-sense Phase 1).

The deterministic core (part-of-day / weekend / season / hemisphere) is fully covered.
The optional sun/holiday enrichments are only asserted to degrade gracefully when their
libraries aren't installed (astral / holidays); on a box with them, semantic_clock just
appends an extra clause.

    python test_timesense.py
"""

import datetime
import importlib.util
import sqlite3
from pathlib import Path

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ts = _load("timesense")

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


def test_part_of_day():
    cases = {2: "the small hours", 6: "early morning", 9: "morning", 13: "midday",
             16: "afternoon", 19: "evening", 23: "night", 0: "the small hours"}
    for h, want in cases.items():
        check(f"hour {h} → {want}", ts.part_of_day(h) == want)


def test_season_and_hemisphere():
    check("January is winter (north)", ts.season(1) == "winter")
    check("July is summer (north)", ts.season(7) == "summer")
    check("October is autumn (north)", ts.season(10) == "autumn")
    check("January is summer (south)", ts.season(1, southern=True) == "summer")
    check("July is winter (south)", ts.season(7, southern=True) == "winter")


def test_day_descriptor():
    check("Monday → start of work week", ts.day_descriptor(0) == "the start of the work week")
    check("Friday → end of work week", ts.day_descriptor(4) == "the end of the work week")
    check("Wednesday → midweek", ts.day_descriptor(2) == "midweek")
    check("Saturday → weekend", ts.day_descriptor(5) == "the weekend")
    check("Sunday → weekend", ts.day_descriptor(6) == "the weekend")


def test_semantic_clock_base():
    s = ts.semantic_clock(datetime.datetime(2026, 6, 27, 19, 30))     # a Saturday
    check("Saturday evening reads naturally",
          s.startswith("It's evening on Saturday — the weekend, summer."))
    s2 = ts.semantic_clock(datetime.datetime(2026, 1, 5, 7, 0))       # a Monday
    check("Monday early winter morning",
          s2.startswith("It's early morning on Monday — the start of the work week, winter."))


def test_southern_inferred_from_latitude():
    s = ts.semantic_clock(datetime.datetime(2026, 1, 15, 14, 0), lat=-33.8, lon=151.2)
    check("negative latitude flips the hemisphere (Jan → summer)", "summer" in s)
    n = ts.semantic_clock(datetime.datetime(2026, 1, 15, 14, 0), lat=51.5, lon=-0.1)
    check("positive latitude stays northern (Jan → winter)", "winter" in n)


def test_optional_enrichments_degrade():
    # With astral/holidays absent, the helpers return "" and semantic_clock is just the base.
    now = datetime.datetime(2026, 6, 27, 19, 30)
    check("sun phrase empty without coords/astral", ts._sun_phrase(now, None, None, None) == "")
    check("holiday phrase empty without a country", ts._holiday_phrase(now, None) == "")
    if not ts._HOLIDAYS:
        check("holiday phrase empty when 'holidays' isn't installed",
              ts._holiday_phrase(now, "GB") == "")
    if not ts._ASTRAL:
        check("sun phrase empty when 'astral' isn't installed",
              ts._sun_phrase(now, "London, UK", None, None) == "")
    # semantic_clock never raises, whatever the inputs
    check("semantic_clock tolerates a bad latitude",
          ts.semantic_clock(now, lat="not-a-number").startswith("It's evening"))


def _usage():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    return ts.UsageLog(db)


def _seed(u, weekday, hour, count, start_day=6):
    """Log `count` sessions on a given weekday at a given hour, on distinct dates."""
    base = datetime.datetime(2026, 1, start_day, hour, 0)        # 2026-01-06 is a Tuesday
    base += datetime.timedelta(days=(weekday - base.weekday()) % 7)
    for i in range(count):
        u.log("session", ts=(base + datetime.timedelta(weeks=i)).timestamp())


def test_usage_silent_until_enough_data():
    u = _usage()
    for _ in range(5):
        u.log("session", ts=datetime.datetime(2026, 1, 6, 19, 0).timestamp())
    check("no rhythm asserted below MIN_EVENTS", u.summary() == "")


def test_usage_evening_rhythm():
    u = _usage()
    # 16 evening sessions across weekdays
    for wd in range(5):
        _seed(u, wd, 19, 4)
    s_evening = u.summary(now=datetime.datetime(2026, 6, 23, 19, 30))   # a Tuesday evening
    check("evening rhythm is detected", "evenings" in s_evening)
    check("a usual time adds no 'unusual' note", "later than" not in s_evening)
    s_late = u.summary(now=datetime.datetime(2026, 6, 24, 3, 0))        # 3am
    check("an off-hour reads as later than usual", "later than you usually talk" in s_late)


def test_usage_weekend_quieter():
    u = _usage()
    for wd in range(5):                       # busy weekday evenings
        _seed(u, wd, 19, 5)
    _seed(u, 5, 19, 1)                         # a single Saturday
    s = u.summary(now=datetime.datetime(2026, 6, 27, 19, 30))           # a Saturday
    check("weekend-quieter note appears on a quiet weekend", "weekends are usually quieter" in s)


def test_usage_no_clear_rhythm():
    u = _usage()
    for hour in range(24):                     # active every hour → no discernible pattern
        u.log("session", ts=datetime.datetime(2026, 1, 6, hour, 0).timestamp())
    check("a flat distribution yields no rhythm line", u.summary() == "")


def test_usage_histograms():
    u = _usage()
    _seed(u, 1, 19, 3)                          # 3 Tuesday-evening sessions
    h = u.histograms()
    check("histogram counts the events", h["n"] == 3)
    check("hour bucket is populated", h["hours"][19] == 3)
    check("weekday bucket is populated", h["wdays"][1] == 3)


def _series(start, step_days, n, hour=18):
    base = start.replace(hour=hour, minute=0, second=0, microsecond=0)
    return [(base + datetime.timedelta(days=step_days * i)).timestamp() for i in range(n)]


def test_detect_weekly_biweekly_interval():
    now = datetime.datetime(2026, 6, 29, 10, 0)
    wk = ts.detect_rhythm(_series(datetime.datetime(2026, 1, 6), 7, 10), now)   # Tuesdays
    check("weekly detected", wk["kind"] == "weekly" and wk["detail"] == "Tuesdays")
    bi = ts.detect_rhythm(_series(datetime.datetime(2026, 1, 6), 14, 8), now)
    check("biweekly detected", bi["kind"] == "biweekly")
    iv = ts.detect_rhythm(_series(datetime.datetime(2026, 1, 1), 9, 9), now)
    check("interval ('every ~9 days') detected",
          iv["kind"] == "interval" and "9 days" in iv["detail"])


def test_detect_monthly():
    import calendar
    now = datetime.datetime(2026, 6, 29, 10, 0)
    firsts = [datetime.datetime(2026, m, 1, 9, 0).timestamp() for m in range(1, 7)]
    dom = ts.detect_rhythm(firsts, now)
    check("monthly-by-date detected", dom["kind"] == "monthly-dom" and "1st" in dom["detail"])
    lastfri = []
    for m in range(1, 7):
        last = max(datetime.date(2026, m, d) for d in range(1, calendar.monthrange(2026, m)[1] + 1)
                   if datetime.date(2026, m, d).weekday() == 4)
        lastfri.append(datetime.datetime(last.year, last.month, last.day, 17).timestamp())
    nth = ts.detect_rhythm(lastfri, now)
    check("monthly nth-weekday ('last Friday') detected — not mislabelled weekly",
          nth["kind"] == "monthly-nth" and nth["detail"] == "the last Friday of the month")
    check("next_expected is in the future", nth["next_expected"] > now.timestamp())


def test_detect_negatives():
    now = datetime.datetime(2026, 6, 29, 10, 0)
    check("too few occurrences → None", ts.detect_rhythm(_series(now, 7, 3), now) is None)
    import random
    random.seed(2)
    noise = [(datetime.datetime(2026, 1, 1) + datetime.timedelta(days=random.randint(0, 180))).timestamp()
             for _ in range(8)]
    r = ts.detect_rhythm(noise, now)
    check("random scatter → no confident rhythm (or low conf)", r is None or r["confidence"] < 0.6)


def test_rhythmstore_upsert_dismiss():
    _, rs = _usage_and_rhythms()
    r = {"kind": "weekly", "period_days": 7, "detail": "Tuesdays", "confidence": 0.9,
         "support": 10, "last_seen": 0, "next_expected": 0}
    rs.upsert("gym", r)
    check("upsert then list shows the rhythm", [x["label"] for x in rs.list()] == ["gym"])
    rs.set_status("gym", "dismissed")
    check("dismissed rhythm leaves the active list", rs.list() == [])
    rs.upsert("gym", r)
    check("a dismissed rhythm is NOT resurrected by re-detection", rs.list() == [])


def _usage_and_rhythms():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    return ts.UsageLog(db), ts.RhythmStore(db)


def test_refresh_contact_and_label():
    u, rs = _usage_and_rhythms()
    for t in _series(datetime.datetime(2026, 1, 4), 7, 12):        # weekly Sunday sessions
        u.log("session", ts=t)
    for t in _series(datetime.datetime(2026, 1, 6), 7, 12):        # weekly Tuesday "gym"
        u.log("activity", label="gym", ts=t)
    now = datetime.datetime(2026, 6, 30, 10, 0)                    # a Tuesday
    n = rs.refresh(u, now=now)
    labels = {x["label"] for x in rs.list()}
    check("refresh detects both the contact cadence and the labelled rhythm",
          n >= 2 and "contact" in labels and "gym" in labels)


def test_relevant_due_and_overdue():
    u, rs = _usage_and_rhythms()
    for t in _series(datetime.datetime(2026, 1, 6), 7, 12):        # weekly Tuesday "gym"
        u.log("activity", label="gym", ts=t)
    tue = datetime.datetime(2026, 6, 30, 10, 0)                    # a Tuesday
    rs.refresh(u, now=tue)
    due = rs.relevant(now=tue)
    check("a due rhythm surfaces around its time", "gym" in due)
    # two weeks later with no new event → overdue
    later = tue + datetime.timedelta(days=14)
    od = rs.relevant(now=later)
    check("an overdue rhythm surfaces as 'longer than usual'",
          "longer than usual" in od and "gym" in od)
    check("nothing due/overdue → empty",
          rs.relevant(now=tue + datetime.timedelta(days=3)) == "")


def main():
    test_part_of_day()
    test_season_and_hemisphere()
    test_day_descriptor()
    test_semantic_clock_base()
    test_southern_inferred_from_latitude()
    test_optional_enrichments_degrade()
    test_usage_silent_until_enough_data()
    test_usage_evening_rhythm()
    test_usage_weekend_quieter()
    test_usage_no_clear_rhythm()
    test_usage_histograms()
    test_detect_weekly_biweekly_interval()
    test_detect_monthly()
    test_detect_negatives()
    test_rhythmstore_upsert_dismiss()
    test_refresh_contact_and_label()
    test_relevant_due_and_overdue()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
