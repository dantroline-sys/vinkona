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

Holders: lm_big has SEVERAL independent holders (the bridge's per-call stream hold and
the research worker's phase hold, possibly in different processes), and one shared file
let any of them delete — or overwrite — another's live hold: the worker finishing its
job used to release the bridge's mid-deliberation lease, and the knowledge-host resumed
verify against a busy 3090.  So each holder now gets its OWN file, `<name>.<holder>.busy`
(the tier is held while ANY of its files is unexpired), and release only ever unlinks the
caller's own — overlap-safe by construction, no coordination, still TTL-crash-safe.
`holder=None` keeps the legacy single `<name>.busy` file, which readers still honour.

Cross-process and cross-app: no shared DB, just tiny files.  Best-effort throughout —
a filesystem hiccup must never break the caller, so every operation swallows OSError.
"""

import contextlib
import os
import re
import time
from pathlib import Path

FAST = "lm_fast"
BIG = "lm_big"
DEFAULT_TTL = 15.0          # seconds a hold stays valid without a refresh

_HOLDER_OK = re.compile(r"[^A-Za-z0-9_-]+")


def control_dir() -> Path:
    """The logs/control directory.  Defaults next to this file (so it resolves to whichever
    checkout is running — dev or the live install); override with VINKONA_CONTROL_DIR."""
    env = os.environ.get("VINKONA_CONTROL_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "logs" / "control"


def _path(name: str, d: Path | None = None, holder: str | None = None) -> Path:
    base = d or control_dir()
    if holder:
        return base / f"{name}.{_HOLDER_OK.sub('-', str(holder))}.busy"
    return base / f"{name}.busy"


def _expired(p: Path, now: float) -> bool:
    """True iff the file exists but its stamp has lapsed (unreadable counts as lapsed)."""
    try:
        return float(p.read_text().strip() or 0) <= now
    except (OSError, ValueError):
        return True


def _holder_files(name: str, d: Path | None = None):
    """Every per-holder lease file for `name` (the legacy `<name>.busy` excluded)."""
    base = d or control_dir()
    try:
        return [p for p in base.iterdir()
                if p.name.startswith(name + ".") and p.name.endswith(".busy")
                and p.name != f"{name}.busy"]
    except OSError:
        return []


def acquire(name: str, *, ttl: float = DEFAULT_TTL, dir: Path | None = None,
            holder: str | None = None) -> None:
    """Take or extend a lease so it stays held for `ttl` seconds.  Idempotent — call it
    again (a keepalive, or once per loop iteration) to refresh a long hold.  Pass a
    stable `holder` id so this hold gets its own file and a sibling's release can never
    drop it (see the module docstring)."""
    p = _path(name, dir, holder)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(repr(time.time() + max(0.01, float(ttl))))
    except OSError:
        pass


refresh = acquire           # same operation, clearer at a keepalive call site


def release(name: str, *, dir: Path | None = None, holder: str | None = None) -> None:
    """Drop OUR hold now (best-effort): only the caller's own file is unlinked, so a
    concurrent holder's live lease survives.  Expired sibling files are swept as
    house-keeping (crashed holders leave them; readers already ignore them).  Even if
    this is missed, the lease expires on its own at the TTL, so a reader never blocks
    forever."""
    try:
        _path(name, dir, holder).unlink()
    except OSError:
        pass
    now = time.time()
    for p in _holder_files(name, dir):
        if _expired(p, now):
            try:
                p.unlink()
            except OSError:
                pass


def is_held(name: str, *, dir: Path | None = None) -> bool:
    """True iff ANY of `name`'s lease files (legacy or per-holder) is unexpired.  This
    is what the knowledge-host calls before each task batch."""
    now = time.time()
    legacy = _path(name, dir)
    try:
        if legacy.exists() and not _expired(legacy, now):
            return True
    except OSError:
        pass
    return any(not _expired(p, now) for p in _holder_files(name, dir))


@contextlib.contextmanager
def held(name: str, *, ttl: float = DEFAULT_TTL, dir: Path | None = None,
         holder: str | None = None):
    """Hold a lease for the duration of a block (e.g. one big-LM call)."""
    acquire(name, ttl=ttl, dir=dir, holder=holder)
    try:
        yield
    finally:
        release(name, dir=dir, holder=holder)
