#!/usr/bin/env python
"""
Tests for lm_lease.py — the per-LM busy leases Vinkona broadcasts so the knowledge-host
yields the contended GPU.  Pure stdlib + a temp dir; no servers.

    python test_lm_lease.py
"""

import importlib.util
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lm_lease = _load("lm_lease")

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}")


def test_acquire_release_isheld():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        check("not held before acquire", not lm_lease.is_held(lm_lease.FAST, dir=d))
        lm_lease.acquire(lm_lease.FAST, ttl=5, dir=d)
        check("held after acquire", lm_lease.is_held(lm_lease.FAST, dir=d))
        check("the lease file lives in the control dir",
              (d / "lm_fast.busy").exists())
        lm_lease.release(lm_lease.FAST, dir=d)
        check("not held after release", not lm_lease.is_held(lm_lease.FAST, dir=d))
        # release of an absent lease is a harmless no-op
        lm_lease.release(lm_lease.FAST, dir=d)
        check("double release doesn't raise", True)


def test_independent_leases():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        lm_lease.acquire(lm_lease.BIG, ttl=5, dir=d)
        check("big held", lm_lease.is_held(lm_lease.BIG, dir=d))
        check("fast independent (not held)", not lm_lease.is_held(lm_lease.FAST, dir=d))
        lm_lease.acquire(lm_lease.FAST, ttl=5, dir=d)
        lm_lease.release(lm_lease.BIG, dir=d)
        check("releasing big leaves fast held",
              lm_lease.is_held(lm_lease.FAST, dir=d) and not lm_lease.is_held(lm_lease.BIG, dir=d))


def test_expiry():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        lm_lease.acquire(lm_lease.BIG, ttl=0.05, dir=d)
        check("held immediately after a short acquire", lm_lease.is_held(lm_lease.BIG, dir=d))
        time.sleep(0.12)
        check("a stale lease reads as not held (crash-safety)",
              not lm_lease.is_held(lm_lease.BIG, dir=d))


def test_refresh_extends():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        lm_lease.acquire(lm_lease.BIG, ttl=0.1, dir=d)
        time.sleep(0.06)
        lm_lease.refresh(lm_lease.BIG, ttl=0.1, dir=d)   # extend before it lapses
        time.sleep(0.06)
        check("refresh keeps a long hold alive past the original ttl",
              lm_lease.is_held(lm_lease.BIG, dir=d))


def test_held_contextmanager():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        with lm_lease.held(lm_lease.BIG, ttl=5, dir=d):
            check("held inside the context", lm_lease.is_held(lm_lease.BIG, dir=d))
        check("released on exit", not lm_lease.is_held(lm_lease.BIG, dir=d))


def test_corrupt_file_is_not_held():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "lm_fast.busy").write_text("not-a-number")
        check("a garbage lease file reads as not held", not lm_lease.is_held(lm_lease.FAST, dir=d))


def main():
    test_acquire_release_isheld()
    test_independent_leases()
    test_expiry()
    test_refresh_extends()
    test_held_contextmanager()
    test_corrupt_file_is_not_held()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
