"""Cooperative GPU leases — Vinkona yields the GPUs to the live assistant.

Vinkona publishes two lease files in her repo's ``logs/control/``:

  * ``lm_fast.busy`` — a live chat is using the fast LM (4090).  Pause the distil
    (first-pass extraction) stage; keep verify running.
  * ``lm_big.busy``  — Vinkona is doing big-LM work (research/briefing/deliberation,
    3090).  Pause the verify/reconcile stage; keep distil running.

A file's contents are a unix expiry timestamp (float).  ``held`` ⇔ the file exists
AND ``float(contents) > time.time()``.  A missing, unparseable, or expired file is
NOT held — we never block on it, and a crashed Vinkona auto-releases within ~15s as the
freshness stamp lapses.  We only READ these files; they are Vinkona's to write/delete.
"""
from __future__ import annotations

import os
import time

FAST = "lm_fast"     # the 4090 (fast extractor)
BIG = "lm_big"       # the 3090 (big verifier / reconciler)

_DEFAULT = "~/Vinkona/logs/control"


def control_dir(cfg=None) -> str:
    """Resolve the lease directory: $VINKONA_CONTROL_DIR > cfg['control_dir'] > default."""
    d = os.environ.get("VINKONA_CONTROL_DIR") or (cfg or {}).get("control_dir") or _DEFAULT
    return os.path.expanduser(d)


def is_held(name: str, cfg=None) -> bool:
    """True iff the named lease is currently held.  Any error => not held (fail open)."""
    path = os.path.join(control_dir(cfg), name + ".busy")
    try:
        with open(path) as fh:
            return float(fh.read().strip()) > time.time()
    except (OSError, ValueError):
        return False
