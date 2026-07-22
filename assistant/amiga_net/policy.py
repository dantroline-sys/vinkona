"""Egress policy: parse egress.toml, match requests to rules, manage leases.

The policy file is DATA (B-13): nothing here knows any destination.  A rule:

    [[rule]]
    name    = "huggingface-weights"
    hosts   = ["huggingface.co", "*.huggingface.co", "*.hf.co"]
    port    = 443
    methods = ["GET", "HEAD"]
    purpose = "download the model weights you asked for"
    ttl_seconds = 7200          # -> a LEASE: nothing until opened, self-closing
    max_uses    = 10000         # requests per open lease
    auth    = "hf_token"        # attach this config secret as a Bearer header

Leases are visible cross-process the lm_lease way: an open lease writes
var/run/egress_lease.<rule>.json (holder pid, opened, expires, uses) and
removes it on close — `status` reads the directory, a crash leaves a stale
file that expires by its own clock.
"""
from __future__ import annotations

import fnmatch
import json
import os
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # the assistant/ app root
POLICY_PATH = ROOT / "egress.toml"
LEASE_DIR = ROOT / "var" / "run"


class Rule:
    def __init__(self, d: dict):
        self.name = str(d.get("name") or "")
        self.hosts = [str(h).lower() for h in (d.get("hosts") or [])]
        self.port = int(d.get("port") or 443)
        self.methods = [m.upper() for m in (d.get("methods") or ["GET"])]
        self.purpose = str(d.get("purpose") or "")
        self.ttl_seconds = float(d.get("ttl_seconds") or 0)
        self.max_uses = int(d.get("max_uses") or 0)
        self.auth = str(d.get("auth") or "")
        self.enabled = bool(d.get("enabled", True))   # a kill switch, not a delete

    @property
    def leased(self) -> bool:
        return self.ttl_seconds > 0 or self.max_uses > 0

    def matches(self, host: str, port: int, method: str) -> bool:
        h = (host or "").lower()
        return (any(fnmatch.fnmatch(h, pat) for pat in self.hosts)
                and port == self.port and method.upper() in self.methods)


def load(path: Path | None = None) -> list[Rule]:
    """The active rules.  A missing or unparseable policy file means NO rules —
    deny-by-default fails closed, never open."""
    p = path or POLICY_PATH
    try:
        data = tomllib.loads(p.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return []
    rules = [Rule(r) for r in data.get("rule") or [] if isinstance(r, dict)]
    return [r for r in rules if r.name and r.hosts]


def find_all(rules: list[Rule], host: str, port: int, method: str) -> list[Rule]:
    return [r for r in rules if r.matches(host, port, method)]


def find(rules: list[Rule], host: str, port: int, method: str) -> Rule | None:
    hits = find_all(rules, host, port, method)
    return hits[0] if hits else None


# ── leases ───────────────────────────────────────────────────────────────────

def _lease_path(rule_name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in rule_name)
    return LEASE_DIR / f"egress_lease.{safe}.json"


def lease_state(rule: Rule) -> dict | None:
    """The live lease for a rule, or None when closed/expired/exhausted."""
    try:
        d = json.loads(_lease_path(rule.name).read_text())
    except (OSError, ValueError):
        return None
    if rule.ttl_seconds and time.time() > float(d.get("expires") or 0):
        return None
    if rule.max_uses and int(d.get("uses") or 0) >= rule.max_uses:
        return None
    return d


def lease_open(rule: Rule, purpose: str) -> dict:
    LEASE_DIR.mkdir(parents=True, exist_ok=True)
    now = time.time()
    d = {"rule": rule.name, "purpose": purpose, "pid": os.getpid(),
         "opened": now, "expires": now + (rule.ttl_seconds or 3600), "uses": 0}
    p = _lease_path(rule.name)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d))
    os.replace(tmp, p)
    return d


def lease_use(rule: Rule) -> None:
    p = _lease_path(rule.name)
    try:
        d = json.loads(p.read_text())
        d["uses"] = int(d.get("uses") or 0) + 1
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(d))
        os.replace(tmp, p)
    except (OSError, ValueError):
        pass


def lease_close(rule_name: str) -> None:
    try:
        _lease_path(rule_name).unlink()
    except OSError:
        pass


def live_leases(rules: list[Rule]) -> list[dict]:
    out = []
    for r in rules:
        d = lease_state(r)
        if d:
            d = dict(d)
            d["remaining_s"] = max(0, int(float(d.get("expires", 0)) - time.time()))
            out.append(d)
    return out


def set_rule_enabled(name: str, on: bool, path: Path | None = None) -> None:
    """Flip one rule's kill switch in egress.toml, in place — comments and the
    rest of the file untouched.  Disabling never deletes: the rule stays
    visible (and re-enableable) in the policy window."""
    import re as _re
    p = path or POLICY_PATH
    lines = p.read_text().splitlines()
    blocks: list[tuple[int, int]] = []
    start = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("["):
            if start is not None:
                blocks.append((start, i))
                start = None
            if s == "[[rule]]":
                start = i
    if start is not None:
        blocks.append((start, len(lines)))
    name_re = _re.compile(r"""^\s*name\s*=\s*["'](?P<v>[^"']*)["']""")
    en_re = _re.compile(r"^(\s*)enabled\s*=.*$")
    for a, b in blocks:
        if not any((m := name_re.match(lines[j])) and m.group("v") == name
                   for j in range(a, b)):
            continue
        for j in range(a, b):
            m = en_re.match(lines[j])
            if m:
                lines[j] = f"{m.group(1)}enabled = {'true' if on else 'false'}"
                break
        else:
            lines[a + 1:a + 1] = [f"enabled = {'true' if on else 'false'}"]
        tmp = p.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + "\n")
        os.replace(tmp, p)
        return
    raise ValueError(f"no rule named '{name}' in {p.name}")
