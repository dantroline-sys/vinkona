#!/usr/bin/env python
"""Pure coordination logic for the 'what is Vinkona doing' status + graceful
preemption (worker_activity.py).  The cascade and the idle worker are separate
processes that agree only through activity.json and a worker_state row, so the
parse/decide rules must stay in lockstep — pin them here.

    python assistant/test_worker_activity.py
"""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("worker_activity",
                                              Path(__file__).parent / "worker_activity.py")
wa = importlib.util.module_from_spec(spec); spec.loader.exec_module(wa)

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}")


STALE = 1800.0

def test_should_yield():
    # a session opened just now → yield idle work
    live = json.dumps(wa.session_record(True, "text", now=1000.0))
    check("live open session → yield", wa.should_yield(live, now=1000.5, open_stale=STALE) is True)
    # open but silent past open_stale → abandoned, don't yield to a ghost
    check("open but stale (abandoned) → don't yield",
          wa.should_yield(live, now=1000.0 + STALE + 1, open_stale=STALE) is False)
    # closed session → don't yield
    closed = json.dumps(wa.session_record(False, None, now=1000.0))
    check("closed session → don't yield", wa.should_yield(closed, now=1001.0, open_stale=STALE) is False)
    # no file ⇒ nothing running ⇒ don't yield
    check("missing activity file → don't yield", wa.should_yield(None, now=1.0, open_stale=STALE) is False)
    check("empty text → don't yield", wa.should_yield("", now=1.0, open_stale=STALE) is False)
    # torn/unreadable file ⇒ fail toward yielding (never fight a possibly-live session)
    check("torn file → yield (fail safe)", wa.should_yield("{not json", now=1.0, open_stale=STALE) is True)


def test_read_session():
    live = json.dumps(wa.session_record(True, "audio", now=500.0))
    s = wa.read_session(live, now=505.0, open_stale=STALE)
    check("live session reads active with its kind", s["active"] is True and s["kind"] == "audio")
    check("age is measured from ts", abs(s["age_s"] - 5.0) < 1e-6)
    stale = wa.read_session(live, now=500.0 + STALE + 10, open_stale=STALE)
    check("stale open session is not 'active' but flagged stale",
          stale["active"] is False and stale["stale"] is True and stale["open"] is True)
    none = wa.read_session(None, now=1.0, open_stale=STALE)
    check("no file → not active, no kind", none["active"] is False and none["kind"] is None)
    torn = wa.read_session("{bad", now=1.0, open_stale=STALE)
    check("torn file → not active, marked unreadable", torn["active"] is False and torn.get("unreadable"))


def test_activity_record_and_labels():
    rec = wa.activity_record("mind_graph", now=42.0)
    check("activity record carries doing + since", rec["doing"] == "mind_graph" and rec["since"] == 42.0)
    check("known task gets its plain-language label",
          rec["label"] == wa.LABELS["mind_graph"] and rec["interruptible"] is True)
    detailed = wa.activity_record("research", now=1.0, detail="how do arrays work")
    check("research label appends the topic detail",
          detailed["label"].endswith("how do arrays work") and "Research" in detailed["label"])
    unknown = wa.activity_record("frobnicate", now=1.0)
    check("unknown task key never yields a blank label", bool(unknown["label"]))
    crit = wa.activity_record("reconcile", now=1.0, interruptible=False)
    check("interruptible flag is honoured", crit["interruptible"] is False)


def test_read_activity_roundtrip():
    txt = json.dumps(wa.activity_record("rss", now=100.0))
    got = wa.read_activity(txt, now=130.0)
    check("read_activity round-trips doing/label", got["doing"] == "rss" and got["label"] == wa.LABELS["rss"])
    check("read_activity adds age_s from since", abs(got["age_s"] - 30.0) < 1e-6)
    check("read_activity(None) is None", wa.read_activity(None, now=1.0) is None)
    check("read_activity(torn) is None", wa.read_activity("{bad", now=1.0) is None)


def main():
    test_should_yield()
    test_read_session()
    test_activity_record_and_labels()
    test_read_activity_roundtrip()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
