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


def test_holder_files():
    """Per-holder leases: each holder gets its own file; the tier is held while ANY
    file is live (July #5 — one shared file let holders clobber each other)."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        lm_lease.acquire(lm_lease.BIG, ttl=5, dir=d, holder="worker-1")
        check("holder file named <name>.<holder>.busy", (d / "lm_big.worker-1.busy").exists())
        check("held via a holder file", lm_lease.is_held(lm_lease.BIG, dir=d))
        check("holder id is sanitised (no path tricks)",
              lm_lease._path(lm_lease.BIG, d, "../evil").name == "lm_big.-evil.busy")
        lm_lease.release(lm_lease.BIG, dir=d, holder="worker-1")
        check("released holder drops the hold", not lm_lease.is_held(lm_lease.BIG, dir=d))


def test_two_holders_overlap():
    """THE July #5 bug: worker finishing its job released the bridge's live
    mid-deliberation lease.  With per-holder files each release is exact."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        lm_lease.acquire(lm_lease.BIG, ttl=5, dir=d, holder="bridge-1-7")
        lm_lease.acquire(lm_lease.BIG, ttl=5, dir=d, holder="worker-1")
        lm_lease.release(lm_lease.BIG, dir=d, holder="worker-1")
        check("worker's release leaves the bridge's live hold standing",
              lm_lease.is_held(lm_lease.BIG, dir=d))
        lm_lease.release(lm_lease.BIG, dir=d, holder="bridge-1-7")
        check("last holder out drops the tier", not lm_lease.is_held(lm_lease.BIG, dir=d))


def test_holder_and_legacy_coexist():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        lm_lease.acquire(lm_lease.BIG, ttl=5, dir=d)                       # legacy caller
        lm_lease.acquire(lm_lease.BIG, ttl=5, dir=d, holder="worker-1")
        lm_lease.release(lm_lease.BIG, dir=d, holder="worker-1")
        check("holder release never unlinks the legacy file",
              lm_lease.is_held(lm_lease.BIG, dir=d))
        lm_lease.release(lm_lease.BIG, dir=d)
        check("legacy release drops the legacy hold", not lm_lease.is_held(lm_lease.BIG, dir=d))


def test_holder_expiry_and_sweep():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        lm_lease.acquire(lm_lease.BIG, ttl=0.05, dir=d, holder="crashed-99")
        time.sleep(0.12)
        check("an expired holder file reads as not held", not lm_lease.is_held(lm_lease.BIG, dir=d))
        (d / "lm_big.garbage.busy").write_text("not-a-number")
        check("a corrupt holder file reads as not held", not lm_lease.is_held(lm_lease.BIG, dir=d))
        lm_lease.release(lm_lease.BIG, dir=d, holder="anyone")             # sweeps the dead
        check("release sweeps expired/corrupt holder files",
              not (d / "lm_big.crashed-99.busy").exists()
              and not (d / "lm_big.garbage.busy").exists())
        # …but never sweeps a LIVE sibling
        lm_lease.acquire(lm_lease.BIG, ttl=5, dir=d, holder="alive")
        lm_lease.release(lm_lease.BIG, dir=d, holder="someone-else")
        check("sweep leaves live siblings alone", lm_lease.is_held(lm_lease.BIG, dir=d))


def test_holder_tiers_independent():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        lm_lease.acquire(lm_lease.FAST, ttl=5, dir=d, holder="cascade-1")
        check("a fast holder never reads as big held",
              lm_lease.is_held(lm_lease.FAST, dir=d) and not lm_lease.is_held(lm_lease.BIG, dir=d))


def test_held_contextmanager_holder():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        with lm_lease.held(lm_lease.BIG, ttl=5, dir=d, holder="ctx-1"):
            check("held(holder=) holds via its own file",
                  (d / "lm_big.ctx-1.busy").exists() and lm_lease.is_held(lm_lease.BIG, dir=d))
        check("held(holder=) releases its own file on exit",
              not lm_lease.is_held(lm_lease.BIG, dir=d))


def main():
    test_acquire_release_isheld()
    test_independent_leases()
    test_expiry()
    test_refresh_extends()
    test_held_contextmanager()
    test_corrupt_file_is_not_held()
    test_holder_files()
    test_two_holders_overlap()
    test_holder_and_legacy_coexist()
    test_holder_expiry_and_sweep()
    test_holder_tiers_independent()
    test_held_contextmanager_holder()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
