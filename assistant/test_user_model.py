"""Tests for user_model — the explicit profile of THIS user (domain fluency,
corrections, communication patterns).  Regression coverage for the counter-reset
bug in set_domain_fluency and the correction bookkeeping."""
import sqlite3

from user_model import UserModelStore


def _store():
    db = sqlite3.connect(":memory:")
    return UserModelStore(db)


def test_set_domain_fluency_preserves_counters():
    um = _store()
    for _ in range(7):
        um.record_domain_interaction("music")
    um.db.execute("UPDATE user_domain_fluency SET correction_count=3 WHERE domain='music'")
    um.set_domain_fluency("music", "expert")            # explicit override
    row = um.get_domain_fluency("music")
    assert row["fluency_level"] == "expert" and row["confidence"] == 1.0
    # the old INSERT OR REPLACE + boolean OR clobbered these to 1 and 0
    assert row["interaction_count"] == 7
    assert row["correction_count"] == 3


def test_set_domain_fluency_creates_fresh_row():
    um = _store()
    um.set_domain_fluency("sailing", "novice")
    row = um.get_domain_fluency("sailing")
    assert row["fluency_level"] == "novice" and row["interaction_count"] == 0


def test_set_domain_fluency_rejects_bad_level():
    um = _store()
    try:
        um.set_domain_fluency("x", "guru")
        assert False, "should raise"
    except ValueError:
        pass


def test_record_correction_bumps_domain_counter():
    um = _store()
    um.record_correction("q", "r", "actually X", domain="music",
                         correction_type="factual_error")
    um.record_correction("q2", "r2", "meant Y", domain="music",
                         correction_type="clarification")
    row = um.get_domain_fluency("music")
    # factual_error creates the fluency row (via record_domain_interaction), then both
    # corrections in the domain are counted
    assert row["correction_count"] == 2
    summary = um.get_correction_summary()
    assert summary["by_domain"]["music"] == 2
    assert summary["by_type"] == {"factual_error": 1, "clarification": 1}


def test_record_correction_rejects_unknown_type():
    um = _store()
    try:
        um.record_correction("q", "r", "c", correction_type="nope")
        assert False, "should raise"
    except ValueError:
        pass


def test_domain_expert_correction_promotes_fluency():
    um = _store()
    um.record_domain_interaction("astronomy")           # starts intermediate
    um.record_correction("q", "r", "I published on this", domain="astronomy",
                         correction_type="domain_expert")
    row = um.get_domain_fluency("astronomy")
    assert row["fluency_level"] == "expert" and row["confidence"] == 0.9


def main():
    import types
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and isinstance(fn, types.FunctionType):
            try:
                fn(); passed += 1; print(f"  ok  {name}")
            except Exception as e:
                failed += 1; print(f"FAIL  {name}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
