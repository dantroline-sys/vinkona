"""Tests for calendar_resolve — the deterministic DateRef resolver (contract §9.1/§9.4)."""
import datetime
import types

import calendar_resolve as cr

ANCHOR = datetime.date(2026, 7, 2)   # Thursday, the contract's canonical anchor
D = datetime.date


# ── §9.1 resolver table (anchor Thursday 2026-07-02) ─────────────────────────────

def test_today_tomorrow_yesterday():
    assert cr.resolve("today", ANCHOR).date == D(2026, 7, 2)
    assert cr.resolve("tomorrow", ANCHOR).date == D(2026, 7, 3)
    assert cr.resolve("yesterday", ANCHOR).date == D(2026, 7, 1)


def test_weekday_this():
    assert cr.resolve("weekday:tue:this", ANCHOR).date == D(2026, 7, 7)   # #3
    assert cr.resolve("weekday:fri:this", ANCHOR).date == D(2026, 7, 3)   # #4 (tomorrow)


def test_weekday_this_when_anchor_is_that_day():
    # #5 — §5.2 today-rule: a mutation on "this Thursday" resolves to TODAY, flagged assumed.
    r = cr.resolve("weekday:thu:this", ANCHOR, is_query=False)
    assert r.date == D(2026, 7, 2) and r.assumed is True and r.policy_note
    # As a query it's today too, but not flagged.
    q = cr.resolve("weekday:thu:this", ANCHOR, is_query=True)
    assert q.date == D(2026, 7, 2) and q.assumed is False


def test_weekday_next_is_following_iso_week():
    assert cr.resolve("weekday:wed:next", ANCHOR).date == D(2026, 7, 8)   # #6
    assert cr.resolve("weekday:fri:next", ANCHOR).date == D(2026, 7, 10)  # #7


def test_relative():
    assert cr.resolve("relative:+3d", ANCHOR).date == D(2026, 7, 5)       # #8
    assert cr.resolve("relative:+2w", ANCHOR).date == D(2026, 7, 16)


def test_explicit_dates():
    assert cr.resolve("explicit:the 14th", ANCHOR).date == D(2026, 7, 14)  # #9 (future this month)
    assert cr.resolve("explicit:july 9", ANCHOR).date == D(2026, 7, 9)     # #11


def test_explicit_dayfirst_and_rollforward():
    # #10 — 1/7 with dayfirst=True is 1 July, which has passed → rolls to next month, assumed.
    r = cr.resolve("explicit:1/7", ANCHOR, dayfirst=True)
    assert r.date == D(2026, 8, 1) and r.assumed is True
    # dayfirst=False flips it to 7 January (past) → rolls a month too.
    r2 = cr.resolve("explicit:1/7", ANCHOR, dayfirst=False)
    assert r2.date.month == 2 and r2.date.day == 7


def test_explicit_word_month_past_rolls_a_year():
    # A word-month that has passed rolls a full year (vs a bare-day/numeric ref → a month).
    r = cr.resolve("explicit:january 5", ANCHOR)          # Jan 5 has passed
    assert r.date == D(2027, 1, 5) and r.assumed is True


def test_explicit_unparseable_raises():
    try:
        cr.resolve("explicit:blursday", ANCHOR)           # #13
        assert False, "expected ResolverError"
    except cr.ResolverError as e:
        assert e.code == "unparseable_date"


def test_unknown_raises():
    for ref in ("unknown", "", "weekday:funday:this", "relative:+xd", "garbage"):
        try:
            cr.resolve(ref, ANCHOR)
            assert False, f"expected ResolverError for {ref!r}"
        except cr.ResolverError:
            pass


def test_range_this_and_next_week():
    assert cr.resolve_range("range:this week", ANCHOR) == (D(2026, 6, 29), D(2026, 7, 5))  # #12
    assert cr.resolve_range("range:next week", ANCHOR) == (D(2026, 7, 6), D(2026, 7, 12))
    # general A..B form
    assert cr.resolve_range("range:today..weekday:sun:this", ANCHOR) == (D(2026, 7, 2), D(2026, 7, 5))


# ── §9.4 grammar / validation parity ─────────────────────────────────────────────

def test_validate_date_ref():
    for good in ("today", "tomorrow", "yesterday", "unknown", "weekday:mon:this",
                 "weekday:sun:next", "relative:+3d", "relative:+2w", "explicit:july 9",
                 "range:this week", "range:today..tomorrow"):
        assert cr.validate_date_ref(good), good
    for bad in ("2026-07-04", "", "weekday:funday:this", "weekday:mon:soon", "relative:3d",
                "relative:+3m", "someday"):
        assert not cr.validate_date_ref(bad), bad


def test_validate_and_resolve_time():
    assert cr.validate_time_ref("15:00") and cr.validate_time_ref("morning")
    assert cr.validate_time_ref("keep") and cr.validate_time_ref("")
    assert not cr.validate_time_ref("25:00") and not cr.validate_time_ref("3pm")
    assert cr.resolve_time("15:00") == (datetime.time(15, 0), False, False)
    assert cr.resolve_time("morning") == (datetime.time(9, 0), False, False)
    assert cr.resolve_time("evening", part_times={"evening": "19:30"}) == (datetime.time(19, 30), False, False)
    assert cr.resolve_time("allday") == (None, True, False)
    assert cr.resolve_time("keep") == (None, False, True)
    assert cr.resolve_time("") == (None, False, False)


def main():
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and isinstance(fn, types.FunctionType):
            try:
                fn()
                passed += 1
                print(f"  ok  {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL  {name}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
