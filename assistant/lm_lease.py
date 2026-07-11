"""
Per-LM busy *leases* — Vinkona broadcasting which local language model she is actively
using, so a strictly-lower-priority background consumer (the standalone knowledge-host)
can yield the contended GPU and get on with whatever work the other one leaves free.

Pecking order is fixed: the live assistant (and her own research) always win; the
knowledge-host is non-time-critical and always defers.  Vinkona never reads the
knowledge-host's state — only the knowledge-host reads these leases and stands down.

Two leases, one per LM tier, as files in logs/control/ (the same control directory
vinkona.sh already uses):

  • lm_fast.busy — held while Vinkona has a live chat/voice session open.  The fast LM
                   (4090) is latency-critical, so the knowledge-host pauses its DISTIL
                   (first-pass extraction) task while this is held.
  • lm_big.busy  — held around Vinkona's big-LM jobs (research synthesis, the planner
                   briefing, deliberation; the 3090).  The knowledge-host pauses its
                   VERIFY (vet/reconcile) task while this is held.

A *lease*, not a flag: the file's content is a unix expiry time, and the holder refreshes
it during long work.  A reader treats it as held only while unexpired — so if Vinkona
crashes mid-hold, the lease goes stale within the TTL and the knowledge-host resumes on
its own.  A stuck file can never halt ingestion forever.

Cross-process and cross-app: no shared DB, just a tiny file.  Best-effort throughout —
a filesystem hiccup must never break the caller, so every operation swallows OSError.
"""

import contextlib
import os
import time
from pathlib import Path

FAST = "lm_fast"
BIG = "lm_big"
DEFAULT_TTL = 15.0          # seconds a hold stays valid without a refresh


def control_dir() -> Path:
    """The logs/control directory.  Defaults next to this file (so it resolves to whichever
    checkout is running — dev or the live install); override with VINKONA_CONTROL_DIR."""
    env = os.environ.get("VINKONA_CONTROL_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "logs" / "control"


def _path(name: str, d: Path | None = None) -> Path:
    return (d or control_dir()) / f"{name}.busy"


def acquire(name: str, *, ttl: float = DEFAULT_TTL, dir: Path | None = None) -> None:
    """Take or extend a lease so it stays held for `ttl` seconds.  Idempotent — call it
    again (a keepalive, or once per loop iteration) to refresh a long hold."""
    p = _path(name, dir)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(repr(time.time() + max(0.01, float(ttl))))
    except OSError:
        pass


refresh = acquire           # same operation, clearer at a keepalive call site


def release(name: str, *, dir: Path | None = None) -> None:
    """Drop a lease now (best-effort).  Even if this is missed, the lease expires on its
    own at the TTL, so a reader never blocks forever."""
    try:
        _path(name, dir).unlink()
    except OSError:
        pass


def is_held(name: str, *, dir: Path | None = None) -> bool:
    """True iff `name`'s lease exists and has not expired.  This is what the knowledge-host
    calls before each task batch."""
    p = _path(name, dir)
    try:
        if not p.exists():
            return False
        return float(p.read_text().strip() or 0) > time.time()
    except (OSError, ValueError):
        return False


@contextlib.contextmanager
def held(name: str, *, ttl: float = DEFAULT_TTL, dir: Path | None = None):
    """Hold a lease for the duration of a block (e.g. one big-LM call)."""
    acquire(name, ttl=ttl, dir=dir)
    try:
        yield
    finally:
        release(name, dir=dir)
