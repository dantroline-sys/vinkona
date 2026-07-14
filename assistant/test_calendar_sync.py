"""Tests for calendar_sync: the loop-safe mirror reconcile + the local instant-retrieval copy."""
import sqlite3
import time

import calendar_sync as cs


# ── markers / notes round-trip ──────────────────────────────────────────────────

def test_marker_round_trip():
    notes = cs.build_notes("dentist — you hate mornings, I gave you a heads-up", "ABC123")
    assert cs.origin_uid_from_notes(notes) == "ABC123"
    assert cs.comment_from_notes(notes) == "dentist — you hate mornings, I gave you a heads-up"


def test_marker_only_no_comment():
    notes = cs.build_notes("", "U1")
    assert cs.origin_uid_from_notes(notes) == "U1"
    assert cs.comment_from_notes(notes) == ""


def test_unmarked_notes_are_not_mirrors():
    assert cs.origin_uid_from_notes("just a normal event note") is None
    assert cs.origin_uid_from_notes("") is None


def test_normalize_tolerates_structured_and_nonstring_fields():
    # A backend that returns objects for start/end (Google-style) and odd types must NOT crash.
    ev = {"id": "H1", "summary": "Vet", "start": {"dateTime": "2026-07-09T14:00:00", "timeZone": "X"},
          "end": {"date": "2026-07-10"}, "location": None, "calendar": {"name": "Work"},
          "notes": {"unexpected": "object"}}
    n = cs.normalize(ev)
    assert n["uid"] == "H1" and n["title"] == "Vet"
    assert n["start"] == "2026-07-09T14:00:00" and n["end"] == "2026-07-10"
    assert n["location"] == "" and n["notes"] == ""      # coerced, not crashed
    # classify over a batch of such events must not raise, and they sort as origins.
    origins, mirrors, own = cs.classify([ev], vinkona_calendar="Vinkona")
    assert len(origins) == 1 and mirrors == {} and own == []


def test_origin_uid_from_notes_tolerates_nonstring():
    assert cs.origin_uid_from_notes({"x": 1}) is None
    assert cs.origin_uid_from_notes(None) is None
    assert cs.comment_from_notes({"x": 1}) == "" and cs.comment_from_notes(None) == ""
    assert isinstance(cs.comment_from_notes(12345), str)   # coerced, never raises


def test_hash_ignores_note_but_tracks_content():
    a = {"title": "Standup", "start": "2026-07-01T09:00", "end": "", "location": "Zoom"}
    b = dict(a)
    assert cs.event_hash(a) == cs.event_hash(b)
    b["start"] = "2026-07-01T10:00"
    assert cs.event_hash(a) != cs.event_hash(b)


def test_hash_ignores_time_and_text_jitter():
    # A re-export that only REFORMATS the same appointment (trailing seconds, case, extra spaces)
    # must not read as a change — otherwise every pull looks like an update.
    a = {"title": "Ward Round", "start": "2026-07-08T07:00:00", "end": "2026-07-08T08:00", "location": "Ward 3"}
    b = {"title": "ward round",  "start": "2026-07-08T07:00:00", "end": "2026-07-08T08:00:00", "location": "Ward  3"}
    assert cs.event_hash(a) == cs.event_hash(b)
    c = dict(a); c["location"] = "Ward 4"                  # a genuine move still registers
    assert cs.event_hash(a) != cs.event_hash(c)


# ── classify / plan ─────────────────────────────────────────────────────────────

def _origin(uid, title, start, cal="Work", notes=""):
    return {"id": uid, "title": title, "start": start, "calendar": cal, "notes": notes}


def test_classify_splits_origins_and_mirrors():
    mirror_notes = cs.build_notes("my note", "W1")
    events = [
        _origin("W1", "Meeting", "2026-07-01T09:00", cal="Work"),
        {"id": "M1", "title": "Meeting", "start": "2026-07-01T09:00",
         "calendar": "Vinkona", "notes": mirror_notes},
        _origin("P1", "Lunch", "2026-07-01T12:00", cal="Personal"),
    ]
    origins, mirrors, own = cs.classify(events, vinkona_calendar="Vinkona")
    assert {o["uid"] for o in origins} == {"W1", "P1"}
    assert set(mirrors) == {"W1"}
    assert mirrors["W1"]["vinkona_id"] == "M1"
    assert mirrors["W1"]["note"] == "my note"
    assert own == []


def test_untagged_vinkona_event_is_own_not_origin_or_mirror():
    # Vinkona's own addition (booked from conversation): in her calendar, no marker.
    events = [{"id": "X1", "title": "Call mum", "start": "2026-07-01T09:00",
               "calendar": "Vinkona", "notes": "no marker here"}]
    origins, mirrors, own = cs.classify(events, vinkona_calendar="Vinkona")
    assert origins == [] and mirrors == {}
    assert len(own) == 1 and own[0]["uid"] == "X1" and own[0]["vinkona_id"] == "X1"
    # it is NEVER pruned (plan only ever touches marked mirrors)
    assert cs.plan_actions(origins, mirrors, prune=True) == []


def test_plan_create_update_skip():
    origins = [
        cs.normalize(_origin("A", "Call", "2026-07-01T09:00")),
        cs.normalize(_origin("B", "Gym", "2026-07-01T18:00")),
    ]
    # A already mirrored & unchanged → skip; B mirrored but moved → update; C new → create.
    origins.append(cs.normalize(_origin("C", "New thing", "2026-07-02T09:00")))
    mirrors = {
        "A": {"vinkona_id": "mA", "hash": cs.event_hash(origins[0]), "note": "n"},
        "B": {"vinkona_id": "mB", "hash": "stale-hash", "note": "n"},
    }
    ops = {a["op"]: a for a in cs.plan_actions(origins, mirrors)}
    assert ops["skip"]["uid"] == "A"
    assert ops["update"]["uid"] == "B" and ops["update"]["vinkona_id"] == "mB"
    assert ops["create"]["uid"] == "C"


def test_plan_prunes_orphan_mirror_but_only_vinkona_owned():
    origins = [cs.normalize(_origin("A", "Call", "2026-07-01T09:00"))]
    mirrors = {
        "A": {"vinkona_id": "mA", "hash": cs.event_hash(origins[0]), "note": ""},
        "GONE": {"vinkona_id": "mGone", "hash": "h", "note": ""},   # origin cancelled
    }
    deletes = [a for a in cs.plan_actions(origins, mirrors, prune=True) if a["op"] == "delete"]
    assert len(deletes) == 1 and deletes[0]["vinkona_id"] == "mGone"
    # prune off → never deletes
    assert not [a for a in cs.plan_actions(origins, mirrors, prune=False) if a["op"] == "delete"]


def test_no_prune_when_origins_empty_failed_read_does_not_wipe():
    # A failed / timed-out external pull returns the Vinkona mirrors but zero origins.  This must
    # NOT be read as "every appointment cancelled" — pruning here would wipe the whole calendar.
    mirrors = {
        "A": {"vinkona_id": "mA", "hash": "h", "note": ""},
        "B": {"vinkona_id": "mB", "hash": "h", "note": ""},
    }
    actions = cs.plan_actions([], mirrors, prune=True)
    assert actions == []                                  # no deletes off an empty read
    # sanity: with even one origin back, normal pruning of the vanished mirror resumes.
    origins = [cs.normalize(_origin("A", "Call", "2026-07-01T09:00"))]
    m2 = {"A": {"vinkona_id": "mA", "hash": cs.event_hash(origins[0]), "note": ""},
          "B": {"vinkona_id": "mB", "hash": "h", "note": ""}}
    deletes = [a for a in cs.plan_actions(origins, m2, prune=True) if a["op"] == "delete"]
    assert [d["vinkona_id"] for d in deletes] == ["mB"]


def test_adopt_matches_unmarked_copy_instead_of_duplicating():
    # An external event whose verbatim copy already sits unmarked in the Vinkona calendar.
    origins = [cs.normalize(_origin("W1", "Team sync", "2026-07-01T09:00", cal="Work"))]
    own = [{"uid": "X1", "vinkona_id": "X1", "title": "Team sync", "start": "2026-07-01T09:00",
            "end": "", "location": "", "calendar": "Vinkona", "notes": ""}]
    actions = cs.plan_actions(origins, {}, own=own, prune=True)
    assert len(actions) == 1 and actions[0]["op"] == "adopt"
    assert actions[0]["vinkona_id"] == "X1" and actions[0]["uid"] == "W1"
    # Without the own list (adopt off) it would instead create a duplicate.
    assert cs.plan_actions(origins, {}, own=None)[0]["op"] == "create"


def test_adopt_only_on_title_and_start_match():
    origins = [cs.normalize(_origin("W1", "Team sync", "2026-07-01T09:00"))]
    own = [{"uid": "X1", "vinkona_id": "X1", "title": "Different thing",
            "start": "2026-07-01T09:00", "notes": ""}]
    # title differs → no adoption, a genuine own entry is left for create
    assert cs.plan_actions(origins, {}, own=own)[0]["op"] == "create"


def test_reconcile_is_idempotent():
    """A second pass after mirroring produces only skips — no duplicate creates."""
    origins_raw = [_origin("A", "Call", "2026-07-01T09:00"),
                   _origin("B", "Gym", "2026-07-01T18:00")]
    origins = [cs.normalize(o) for o in origins_raw]
    # Simulate the mirrors written by the first pass.
    mirror_events = []
    for o in origins:
        mirror_events.append({"id": "m" + o["uid"], "title": o["title"], "start": o["start"],
                              "calendar": "Vinkona", "notes": cs.build_notes("note", o["uid"])})
    all_events = origins_raw + mirror_events
    origins2, mirrors2, _own = cs.classify(all_events, "Vinkona")
    actions = cs.plan_actions(origins2, mirrors2)
    assert {a["op"] for a in actions} == {"skip"}


def test_uid_churn_unchanged_event_skips_no_write():
    # Hosportal regenerates a fresh UID on every export.  The SAME appointment arrives under a new
    # uid while the mirror still carries the OLD one → recognise it by content and SKIP (no write),
    # never delete-the-mirror + create-a-duplicate.
    mirror = {"id": "mHP", "title": "Ward round", "start": "2026-07-08T07:00",
              "calendar": "Vinkona", "notes": cs.build_notes("early start", "HP-1000")}
    origin_new = _origin("HP-2000", "Ward round", "2026-07-08T07:00")   # same event, churned uid
    origins, mirrors, _own = cs.classify([origin_new, mirror], "Vinkona")
    actions = cs.plan_actions(origins, mirrors, prune=True)
    assert [a["op"] for a in actions] == ["skip"]         # no create, no delete despite the new uid
    assert actions[0]["vinkona_id"] == "mHP" and actions[0]["note"] == "early start"


def test_uid_churn_touches_only_the_changed_shift_not_the_whole_calendar():
    # THE reported bug: one shift changes, Hosportal re-exports with ALL uids churned, and a
    # UID-only reconcile deletes every mirror and recreates every one (calendar wiped + rewritten).
    # With content matching, the unchanged shifts SKIP and only the moved one is create/delete.
    prev = [("Ward round", "2026-07-08T07:00"), ("Clinic", "2026-07-08T13:00"),
            ("On call", "2026-07-09T18:00")]
    mirror_events = [{"id": f"m{i}", "title": t, "start": s, "calendar": "Vinkona",
                      "notes": cs.build_notes("", f"OLD-{i}")} for i, (t, s) in enumerate(prev)]
    now = [("Ward round", "2026-07-08T07:00"), ("Clinic", "2026-07-08T14:00"),   # Clinic moved
           ("On call", "2026-07-09T18:00")]
    origin_events = [_origin(f"NEW-{i}", t, s) for i, (t, s) in enumerate(now)]
    origins, mirrors, _own = cs.classify(origin_events + mirror_events, "Vinkona")
    ops = [a["op"] for a in cs.plan_actions(origins, mirrors, prune=True)]
    assert ops.count("skip") == 2                         # two unchanged shifts: untouched
    assert ops.count("create") == 1 and ops.count("delete") == 1   # only the moved shift
    assert "update" not in ops                            # (a same-day move is create+delete, not wipe)


# ── fold_mirrors (live-scan dedupe) ──────────────────────────────────────────────

def test_fold_mirrors_collapses_to_one_and_keeps_note():
    events = [
        _origin("W1", "Meeting", "2026-07-01T09:00", cal="Work"),
        {"id": "M1", "title": "Meeting", "start": "2026-07-01T09:00",
         "calendar": "Vinkona", "notes": cs.build_notes("heads-up note", "W1")},
    ]
    folded = cs.fold_mirrors(events, "Vinkona")
    assert len(folded) == 1
    assert folded[0]["uid"] == "W1" and folded[0]["note"] == "heads-up note"


def test_fold_mirrors_noop_without_markers():
    events = [_origin("W1", "Meeting", "2026-07-01T09:00"),
              _origin("W2", "Lunch", "2026-07-01T12:00")]
    folded = cs.fold_mirrors(events, "Vinkona")
    assert len(folded) == 2


# ── to_epoch ─────────────────────────────────────────────────────────────────────

def test_to_epoch_parses_and_fails_soft():
    assert cs.to_epoch("2026-07-01T09:00:00Z") is not None
    assert cs.to_epoch("2026-07-01T09:00") is not None
    assert cs.to_epoch("not a date") is None
    assert cs.to_epoch("") is None


# ── CalendarStore ────────────────────────────────────────────────────────────────

def _store():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    return cs.CalendarStore(db)


def _row(uid, title, start_ts, note=""):
    return {"uid": uid, "vinkona_id": "m" + uid, "title": title, "start": "", "end": "",
            "start_ts": start_ts, "location": "", "source": "Work", "note": note,
            "hash": "h", "synced_at": time.time()}


def test_store_replace_all_and_upcoming():
    st = _store()
    now = time.time()
    st.replace_all([
        _row("A", "Past", now - 7200),
        _row("B", "Soon", now + 600, note="don't be late"),
        _row("C", "Later", now + 5 * 86400),
    ])
    up = st.upcoming(now=now, horizon_days=1)
    assert [e["uid"] for e in up] == ["B"]              # past dropped, C beyond horizon
    assert up[0]["note"] == "don't be late"
    # replace_all truly replaces
    st.replace_all([_row("Z", "Only", now + 100)])
    assert [e["uid"] for e in st.all()] == ["Z"]


def test_store_summary():
    st = _store()
    now = time.time()
    st.replace_all([_row("B", "Dentist", now + 3600, note="you booked the early slot")])
    s = st.summary(now=now, horizon_days=1)
    assert "Dentist" in s and "you booked the early slot" in s


def test_store_empty_summary():
    assert _store().summary() == ""


def test_store_distinguishes_self_authored_from_mirrored():
    st = _store()
    now = time.time()
    mirror = _row("M1", "Work meeting", now + 3600); mirror["source"] = "Work"
    own = _row("self:O1", "Call mum", now + 7200); own["source"] = "self"; own["self_authored"] = 1
    st.replace_all([mirror, own])
    rows = {r["uid"]: r for r in st.all()}
    assert rows["M1"]["self_authored"] == 0 and rows["self:O1"]["self_authored"] == 1
    s = st.summary(now=now, horizon_days=1)
    assert "you asked me to add this" in s          # the own event is flagged
    assert "from your Work calendar" in s           # the mirror names its source


def test_store_migrates_old_cache_without_self_authored_column():
    import sqlite3
    db = sqlite3.connect(":memory:"); db.row_factory = sqlite3.Row
    # Old-schema table (no self_authored column).
    db.executescript('CREATE TABLE calendar_events (uid TEXT PRIMARY KEY, vinkona_id TEXT, '
                     'title TEXT, start TEXT, "end" TEXT, start_ts REAL, location TEXT, '
                     'source TEXT, note TEXT, hash TEXT, synced_at REAL);')
    st = cs.CalendarStore(db)                        # _init_db should ALTER in the new column
    st.replace_all([_row("A", "X", time.time() + 60)])
    assert "self_authored" in st.all()[0]


# ── legacy "Amiga" data (the pre-rename assistant) ─────────────────────────────

_LEGACY_NOTES = "my old note\n\n— kept in sync by Amiga [amiga-mirror:W1] —"


def test_legacy_amiga_marker_is_recognised_and_stripped():
    # Old mirrors on a live calendar still carry the pre-rename marker: it must parse as a
    # mirror (or the sync duplicates instead of updating) and the marker line must strip
    # (or "kept in sync by Amiga" leaks into visible notes).
    assert cs.origin_uid_from_notes(_LEGACY_NOTES) == "W1"
    assert cs.comment_from_notes(_LEGACY_NOTES) == "my old note"


def test_classify_flags_legacy_marker():
    events = [
        _origin("W1", "Meeting", "2026-07-01T09:00", cal="Work"),
        {"id": "M1", "title": "Meeting", "start": "2026-07-01T09:00",
         "calendar": "Vinkona", "notes": _LEGACY_NOTES},
    ]
    origins, mirrors, own = cs.classify(events, vinkona_calendar="Vinkona")
    assert set(mirrors) == {"W1"} and own == []
    assert mirrors["W1"]["legacy_marker"] is True
    assert mirrors["W1"]["note"] == "my old note"
    # …and a current-marker mirror is NOT flagged
    events[1]["notes"] = cs.build_notes("my note", "W1")
    _, mirrors2, _ = cs.classify(events, vinkona_calendar="Vinkona")
    assert mirrors2["W1"]["legacy_marker"] is False


def test_classify_treats_legacy_amiga_calendar_as_own():
    # The user's own calendar may still be NAMED "Amiga" — its unmarked events are hers
    # (own), never foreign origins to re-mirror.
    events = [{"id": "X1", "title": "Call mum", "start": "2026-07-01T09:00",
               "calendar": "Amiga", "notes": ""}]
    origins, mirrors, own = cs.classify(events, vinkona_calendar=["Vinkona", "Amiga"])
    assert origins == [] and mirrors == {}
    assert len(own) == 1 and own[0]["uid"] == "X1"
    # without the alias the same event is misread as an origin (the pre-fix bug)
    origins, _, own = cs.classify(events, vinkona_calendar="Vinkona")
    assert len(origins) == 1 and own == []


def test_plan_upgrades_unchanged_legacy_mirror_in_place():
    # Content unchanged, but the mirror still wears the old marker → ONE update rewrites
    # the notes in the new form (comment kept); a current-marker mirror still skips.
    origin = cs.normalize(_origin("W1", "Meeting", "2026-07-01T09:00"))
    mirrors = {"W1": {"vinkona_id": "mA", "hash": cs.event_hash(origin),
                      "note": "my old note", "legacy_marker": True}}
    ops = cs.plan_actions([origin], mirrors)
    assert [a["op"] for a in ops] == ["update"]
    assert ops[0]["note"] == "my old note" and ops[0]["vinkona_id"] == "mA"
    mirrors["W1"]["legacy_marker"] = False
    assert [a["op"] for a in cs.plan_actions([origin], mirrors)] == ["skip"]


def test_plan_prunes_legacy_orphan():
    # A legacy-marked mirror whose origin was cancelled is prunable like any other —
    # before the fix it was invisible and lingered forever.
    events = [
        {"id": "A", "title": "Call", "start": "2026-07-01T09:00", "calendar": "Work",
         "notes": ""},
        {"id": "mGone", "title": "Old thing", "start": "2026-06-01T09:00",
         "calendar": "Vinkona",
         "notes": "bye\n\n— kept in sync by Amiga [amiga-mirror:GONE] —"},
    ]
    origins, mirrors, own = cs.classify(events, vinkona_calendar="Vinkona")
    deletes = [a for a in cs.plan_actions(origins, mirrors, prune=True) if a["op"] == "delete"]
    assert len(deletes) == 1 and deletes[0]["vinkona_id"] == "mGone"


def test_fold_mirrors_folds_legacy_mirror():
    # The live scan must collapse a legacy mirror onto its origin (no double-count) and
    # surface the note WITHOUT any of the old marker text.
    events = [
        _origin("W1", "Meeting", "2026-07-01T09:00", cal="Work"),
        {"id": "M1", "title": "Meeting", "start": "2026-07-01T09:00",
         "calendar": "Vinkona", "notes": _LEGACY_NOTES},
    ]
    folded = cs.fold_mirrors(events, "Vinkona")
    assert len(folded) == 1
    assert folded[0]["note"] == "my old note"
    assert "Amiga" not in folded[0]["note"] and "amiga-mirror" not in folded[0]["note"]


def main():
    import types
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
