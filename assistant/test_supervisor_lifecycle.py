#!/usr/bin/env python
"""Tests for the supervisor's July #6 lifecycle fixes — pure stdlib, no services.

  1. zombie reaping: stops of already-dead (unreaped) children return fast
     instead of sitting out the full GRACE_S and SIGKILLing a zombie;
  2. pidfile exclusivity: the flock claim admits exactly one supervisor
     (the old check-then-write let two racing starts boot two stacks);
  3. svc_check ports: health probes derive the port from the tier's URL
     (cfg[tier]["port"] never existed, so a re-ported LM always read dead).

    python test_supervisor_lifecycle.py
"""
import importlib.util
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sup = _load("supervisor")

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


def test_kill_group_reaps_zombie():
    """A child that already exited but was never poll()ed is a zombie: signal-0
    says alive, so the old loop burned the whole GRACE_S on it."""
    p = subprocess.Popen(["true"], start_new_session=True)
    time.sleep(0.3)                            # let it exit; do NOT poll — zombie
    t0 = time.time()
    sup.kill_group(p.pid, p)
    took = time.time() - t0
    p.poll()                                   # reap if kill_group's early return skipped it
    check(f"kill_group returns fast on a zombie child ({took:.2f}s)", took < 3.0)
    check("the zombie is reaped", p.returncode is not None)


def test_kill_group_terms_live_child():
    p = subprocess.Popen(["sleep", "30"], start_new_session=True)
    t0 = time.time()
    sup.kill_group(p.pid, p)
    took = time.time() - t0
    check(f"kill_group TERMs a live child promptly ({took:.2f}s)", took < 3.0)
    check("the child is reaped after TERM", p.poll() is not None)


def test_stop_all_mixed():
    """stop_all with one unreaped-dead child and one live child: both gone,
    both reaped, and nowhere near services×GRACE_S."""
    with tempfile.TemporaryDirectory() as d:
        old_state = sup.STATE
        sup.STATE = Path(d) / "state.json"
        try:
            s = sup.Supervisor()
            s.svcs = []
            dead = subprocess.Popen(["true"], start_new_session=True)
            live = subprocess.Popen(["sleep", "30"], start_new_session=True)
            time.sleep(0.3)
            s.children = {"dead": dead, "live": live}
            t0 = time.time()
            s.stop_all()
            took = time.time() - t0
            check(f"stop_all returns fast ({took:.2f}s)", took < 4.0)
            check("both children reaped", dead.poll() is not None and live.poll() is not None)
            check("children dict cleared", not s.children)
        finally:
            sup.STATE = old_state


def test_pidfile_claim_exclusive():
    with tempfile.TemporaryDirectory() as d:
        old_ctrl, old_pidfile = sup.CTRL, sup.PIDFILE
        sup.CTRL, sup.PIDFILE = Path(d), Path(d) / "supervisor.pid"
        try:
            f1, other1 = sup._claim_pidfile()
            check("first claim wins", f1 is not None and other1 == "")
            check("pidfile carries our pid", sup.PIDFILE.read_text().strip().isdigit())
            f2, other2 = sup._claim_pidfile()
            check("second claim is refused while the first lives", f2 is None)
            check("the refusal names the holder", other2.strip().isdigit())
            f1.close()                          # first supervisor dies → lock drops
            f3, _ = sup._claim_pidfile()
            check("claim succeeds again after the holder is gone", f3 is not None)
            if f3:
                f3.close()
        finally:
            sup.CTRL, sup.PIDFILE = old_ctrl, old_pidfile


def test_svc_check_ports_from_urls():
    def line(name, cfg):
        return sup.svc_check(name, cfg, {}, True)
    check("fast_lm port comes from its url",
          ":19999" in line("fast_lm", {"fast_lm": {"url": "http://127.0.0.1:19999"}}))
    check("embed maps to the embed_lm block",
          ":21437" in line("embed", {"embed_lm": {"url": "http://127.0.0.1:21437"}}))
    check("big_lm2 inherits big_lm's url",
          ":18888" in line("big_lm2", {"big_lm": {"url": "http://127.0.0.1:18888"}}))
    check("big_lm2's own url overrides the inherited one",
          ":18890" in line("big_lm2", {"big_lm": {"url": "http://127.0.0.1:18888"},
                                       "big_lm2": {"url": "http://127.0.0.1:18890"}}))
    check("no url falls back to the tier default",
          ":11435" in line("fast_lm", {}))
    check("an unparseable url falls back to the tier default",
          ":11438" in line("big_lm", {"big_lm": {"url": "http://127.0.0.1:notaport"}}))


def main():
    test_kill_group_reaps_zombie()
    test_kill_group_terms_live_child()
    test_stop_all_mixed()
    test_pidfile_claim_exclusive()
    test_svc_check_ports_from_urls()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
