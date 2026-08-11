#!/usr/bin/env python
"""The panel's 'what is Vinkona doing' aggregator (config_server.MemoryAdmin.activity_status).

Three processes meet here: the CASCADE writes activity.json, the idle WORKER writes a
worker_state 'activity' row, and the CONFIG server (this) only reads both and resolves one
headline for the UIs.  If the shapes drift the panel shows the wrong thing, so pin the
contract — precedence (a live chat wins), the plain-language headline, and the fail-soft
empties.

    python assistant/test_activity_status.py
"""
import importlib.util
import json
import sqlite3
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


cs = _load("config_server")
wa = _load("worker_activity")

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}")


def _cfg(tmp: Path, db: str):
    return {"memory": {"db_path": db}, "embed_lm": {},
            "config_server": {"trace_path": str(tmp / "trace.jsonl")},
            "research": {"idle": {"open_stale_s": 1800}}}


def _db_with(tmp: Path, rows: dict) -> str:
    db = str(tmp / "mem.db")
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE worker_state (key TEXT PRIMARY KEY, value TEXT)")
    for k, v in rows.items():
        c.execute("INSERT INTO worker_state(key,value) VALUES(?,?)", (k, v))
    c.commit(); c.close()
    return db


def _write_session(tmp: Path, open_, kind, ts):
    (tmp / "activity.json").write_text(json.dumps(wa.session_record(open_, kind, ts)))


def test_live_chat_wins():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # worker says it's distilling, but a live text chat is open → chat wins.
        db = _db_with(tmp, {"activity": json.dumps(wa.activity_record("mind_graph", time.time()))})
        _write_session(tmp, True, "text", time.time())
        st = cs.MemoryAdmin(_cfg(tmp, db)).activity_status(_cfg(tmp, db))
        check("a live chat takes precedence over idle work", st["doing"] == "chatting")
        check("chat headline names the modality", st["headline"] == "In a text conversation")
        check("a live chat is not interruptible", st["interruptible"] is False)
        check("session sub-view is active with its kind",
              st["session"]["active"] and st["session"]["kind"] == "text")


def test_worker_task_when_no_chat():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        db = _db_with(tmp, {"activity": json.dumps(wa.activity_record("rss", time.time()))})
        _write_session(tmp, False, None, time.time())        # session closed
        st = cs.MemoryAdmin(_cfg(tmp, db)).activity_status(_cfg(tmp, db))
        check("with no chat, the worker's task is the headline", st["doing"] == "rss")
        check("headline is the task's plain-language label", st["headline"] == wa.LABELS["rss"])
        check("a background task reports interruptible", st["interruptible"] is True)


def test_stale_session_is_not_a_chat():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        db = _db_with(tmp, {"activity": json.dumps(wa.activity_record("idle", time.time()))})
        _write_session(tmp, True, "audio", time.time() - 4000)   # 'open' but long silent → ghost
        st = cs.MemoryAdmin(_cfg(tmp, db)).activity_status(_cfg(tmp, db))
        check("a stale 'open' heartbeat is not treated as a live chat", st["doing"] != "chatting")
        check("stale session falls through to idle", st["doing"] == "idle")


def test_paused_reads_as_resting():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        db = _db_with(tmp, {"idle_override": "paused",
                            "activity": json.dumps(wa.activity_record("rss", time.time()))})
        _write_session(tmp, False, None, time.time())
        st = cs.MemoryAdmin(_cfg(tmp, db)).activity_status(_cfg(tmp, db))
        check("a manual pause reads as resting, not the mid-flight task", st["doing"] == "paused")


def test_missing_everything_is_idle_not_error():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        db = str(tmp / "absent.db")                          # never created
        st = cs.MemoryAdmin(_cfg(tmp, db)).activity_status(_cfg(tmp, db))
        check("no db + no heartbeat → idle, no error", st["doing"] == "idle")
        check("idle headline is friendly", "Idle" in st["headline"])
        check("session sub-view is inactive", st["session"]["active"] is False)


def main():
    test_live_chat_wins()
    test_worker_task_when_no_chat()
    test_stale_session_is_not_a_chat()
    test_paused_reads_as_resting()
    test_missing_everything_is_idle_not_error()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
