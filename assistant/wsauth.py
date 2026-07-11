"""
Cascade WebSocket access token — a human-typable shared secret a client must present
before the server will talk to it.

The cascade server listens on the network so the phone can reach it, but the WS itself
had no client authentication: anyone who could reach the port could converse with the
assistant and pull the user's private memories.  This closes that hole with a small
pre-shared key the user copies into the client once.

Design:
  • The server generates the token on first run and persists it to a file the user can
    read (e.g. `config/ws_token.txt`); it survives restarts so the client keeps working.
  • The token is Crockford base32 (no 0/O, 1/I/L/U — nothing ambiguous to read or type),
    grouped XXXX-XXXX-XXXX-XXXX.  16 symbols ⇒ 80 bits of entropy: trivial to type, far
    too large to guess.
  • Comparison is case-insensitive and ignores grouping/spaces, and uses a constant-time
    compare so a network attacker can't time their way in.

Verification happens in-band (the client's first WS frame), so it works for the phone and
for the browser text-chat page alike (browsers can't set auth headers).
"""

import hmac
import os
import re
from pathlib import Path

# Crockford base32 minus the visually ambiguous letters — safe to read aloud and type.
_ALPHABET = "23456789ABCDEFGHJKMNPQRSTVWXYZ"


def generate_token(groups: int = 4, group_len: int = 4) -> str:
    """A fresh human-typable token, e.g. 'K7M2-9PQR-4XYZ-AB3C' (default 80 bits)."""
    n = groups * group_len
    raw = os.urandom(n)
    chars = [_ALPHABET[b % len(_ALPHABET)] for b in raw]
    return "-".join("".join(chars[i:i + group_len]) for i in range(0, n, group_len))


def normalize(token: str) -> str:
    """Canonical form for comparison: uppercase, only the allowed alphabet (drop hyphens,
    spaces, and anything else)."""
    if not token:
        return ""
    return re.sub(rf"[^{_ALPHABET}]", "", str(token).upper())


def verify(provided: str, actual: str) -> bool:
    """Constant-time check that `provided` matches `actual`, ignoring case/grouping."""
    a, b = normalize(provided), normalize(actual)
    if not a or not b:
        return False
    return hmac.compare_digest(a, b)


def load_or_create(path: str) -> str:
    """Return the persisted token at `path`, generating and writing it (mode 600) if it
    isn't there yet.  Falls back to an in-memory token if the file can't be written, so a
    locked-down filesystem degrades to 'auth on, but you must read the token from the log'
    rather than failing open."""
    p = Path(path)
    try:
        existing = normalize(p.read_text())
        if existing:
            return p.read_text().strip()
    except Exception:
        pass
    tok = generate_token()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(tok + "\n")
        os.chmod(p, 0o600)
    except Exception:
        pass
    return tok
