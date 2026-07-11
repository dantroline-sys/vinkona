"""Tests for spoken_time: machine date/time -> words a TTS engine reads naturally."""
import datetime
import types

import spoken_time as st


def test_num_word():
    assert st._num_word(0) == "zero"
    assert st._num_word(5) == "five"
    assert st._num_word(15) == "fifteen"
    assert st._num_word(30) == "thirty"
    assert st._num_word(45) == "forty-five"
    assert st._num_word(59) == "fifty-nine"


def test_ordinal_word():
    assert st._ordinal_word(1) == "first"
    assert st._ordinal_word(4) == "fourth"
    assert st._ordinal_word(11) == "eleventh"
    assert st._ordinal_word(12) == "twelfth"
    assert st._ordinal_word(20) == "twentieth"
    assert st._ordinal_word(21) == "twenty-first"
    assert st._ordinal_word(30) == "thirtieth"
    assert st._ordinal_word(31) == "thirty-first"


def test_part_of_day():
    assert st._part_of_day(7) == "morning"
    assert st._part_of_day(13) == "afternoon"
    assert st._part_of_day(19) == "evening"
    assert st._part_of_day(23) == "night"


def test_clock_word():
    d = datetime.datetime
    assert st._clock_word(d(2026, 7, 4, 7, 0)) == "seven"
    assert st._clock_word(d(2026, 7, 4, 7, 0), sharp_oclock=True) == "seven o'clock"
    assert st._clock_word(d(2026, 7, 4, 13, 30)) == "one thirty"
    assert st._clock_word(d(2026, 7, 4, 9, 5)) == "nine oh five"
    assert st._clock_word(d(2026, 7, 4, 0, 0), sharp_oclock=True) == "twelve o'clock"  # midnight
    assert st._clock_word(d(2026, 7, 4, 12, 15)) == "twelve fifteen"                    # noon-ish


def test_parse_variants():
    dt, has = st.parse("2026-07-04T07:00")
    assert dt == datetime.datetime(2026, 7, 4, 7, 0) and has is True
    dt, has = st.parse("2026-07-04T13:30:00")
    assert dt == datetime.datetime(2026, 7, 4, 13, 30) and has is True
    dt, has = st.parse("2026-07-04T07:00:00Z")
    assert dt is not None and has is True
    dt, has = st.parse("2026-07-04")
    assert dt == datetime.datetime(2026, 7, 4, 0, 0) and has is False   # date only
    dt, has = st.parse("not a date")
    assert dt is None and has is False
    dt, has = st.parse("")
    assert dt is None and has is False


def test_spoken_when_range_crossing_noon():
    # The motivating example: Sat 04 Jul 07:00-13:30.
    assert st.spoken_when("2026-07-04T07:00", "2026-07-04T13:30") == (
        "Saturday the fourth of July, from seven in the morning to one thirty in the afternoon")


def test_spoken_when_range_same_part_of_day():
    assert st.spoken_when("2026-07-04T07:00", "2026-07-04T09:30") == (
        "Saturday the fourth of July, from seven to nine thirty in the morning")


def test_spoken_when_single_time():
    assert st.spoken_when("2026-07-04T07:00") == (
        "Saturday the fourth of July, at seven o'clock in the morning")
    # An end equal to the start is treated as a single time, not a zero-length range.
    assert st.spoken_when("2026-07-04T07:00", "2026-07-04T07:00") == (
        "Saturday the fourth of July, at seven o'clock in the morning")


def test_spoken_when_all_day():
    assert st.spoken_when("2026-07-04") == "Saturday the fourth of July"


def test_spoken_when_unparseable_returns_none():
    # Caller keeps its own raw-prose fallback when start won't parse.
    assert st.spoken_when("Sat 04 Jul 07:00-13:30") is None
    assert st.spoken_when("") is None


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
