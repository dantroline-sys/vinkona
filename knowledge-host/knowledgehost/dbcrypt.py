"""SQLCipher encryption at rest for sensitive bundles (spec §16.6).

Scope is deliberate: encrypt the **overlay / sensitive** bundles (the user's own
cards, anything personal); leave the large non-private base **clear** — more
private *and* faster than encrypting everything.  The threat this addresses is a
lost / stolen / seized / backed-up disk: a flagged bundle is ciphertext at rest.
It is **at-rest only** — a running, unlocked server necessarily holds decrypted
pages in RAM.  Necessary, not sufficient.

Guarded by design.  Without a SQLCipher driver installed this reports
``available() == False`` and any caller that *asked* to encrypt **fails loud** —
we never silently write plaintext for a bundle the operator marked sensitive.
The plain path (``encrypted=False``) is just ``sqlite3.connect`` and is
unchanged/untouched for the base and for anyone not using this feature.

Key never lives in the config file.  It comes from ``$KNOWLEDGEHOST_DB_KEY`` or a
``db_key_file`` (mode-600); on the run box the right home is the OS keystore /
TPM (unlocks with device login, key never in the app) — a thin hook to add there.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

_DRIVERS = ("sqlcipher3", "pysqlcipher3.dbapi2")


def _driver():
    for mod in _DRIVERS:
        try:
            return __import__(mod, fromlist=["connect"])
        except Exception:                           # not installed — fine, base is clear
            continue
    return None


def available() -> bool:
    """Is a SQLCipher driver importable on this box?"""
    return _driver() is not None


def key_for(cfg: dict) -> str | None:
    """Resolve the at-rest key: env var wins, then a key file.  Returns None if
    neither is set (callers that need it then fail loud)."""
    k = os.environ.get("KNOWLEDGEHOST_DB_KEY")
    if k:
        return k
    kf = (cfg or {}).get("db_key_file") or ""
    if kf:
        p = Path(kf).expanduser()
        if p.exists():
            return p.read_text().strip()
    return None


def connect(path: str, *, encrypted: bool = False, key: str | None = None, **kw):
    """Open ``path``.  ``encrypted=False`` ⇒ ordinary ``sqlite3`` (the base path,
    byte-for-byte as before).  ``encrypted=True`` ⇒ SQLCipher with ``PRAGMA key``
    applied before any other statement; raises if the driver or key is missing."""
    if not encrypted:
        return sqlite3.connect(path, **kw)
    drv = _driver()
    if drv is None:
        raise RuntimeError(
            "encrypted bundle requested but no SQLCipher driver is installed "
            "(pip install sqlcipher3-binary) — refusing to write plaintext")
    if not key:
        raise RuntimeError(
            "encrypted bundle requested but no key found "
            "($KNOWLEDGEHOST_DB_KEY or db_key_file)")
    con = drv.connect(path, **kw)
    # PRAGMA key must precede every other operation on the connection.
    con.execute("PRAGMA key = ?", (key,))
    # Force the cipher to engage now so a wrong key / non-cipher file fails here
    # (loud) rather than deep in a later query.
    con.execute("SELECT count(*) FROM sqlite_master")
    return con


def bundle_is_encrypted(cfg: dict, bundle: str | None) -> bool:
    """Is this bundle name flagged for at-rest encryption?"""
    if not bundle:
        return False
    return bundle in set((cfg or {}).get("encrypted_bundles") or [])
