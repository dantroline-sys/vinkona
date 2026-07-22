"""The append-only egress audit log (B-10): one JSON line per decision.

Fields: ts (UTC ISO), component, purpose, host, port, rule ('-' when none
matched), verdict (ALLOWED / DENIED / LEASE_OPEN / LEASE_CLOSE / AUTH_REJECT /
POSTURE), bytes_out, bytes_in.  Never request or response bodies — the log
says WHO talked to WHOM and WHY, not what was said.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # the assistant/ app root
LOG_PATH = ROOT / "var" / "log" / "egress.jsonl"

VERDICTS = ("ALLOWED", "DENIED", "LEASE_OPEN", "LEASE_CLOSE", "AUTH_REJECT",
            "POSTURE", "POLICY")   # POLICY = an operator changed the rules/leases


def component() -> str:
    return os.environ.get("AMIGA_COMPONENT", "vinkona")


def write(verdict: str, *, purpose: str = "", host: str = "", port: int = 0,
          rule: str = "-", bytes_out: int = 0, bytes_in: int = 0,
          detail: str = "", path: Path | None = None) -> dict:
    assert verdict in VERDICTS, verdict
    ev = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "component": component(), "purpose": purpose[:200],
          "host": host, "port": port, "rule": rule, "verdict": verdict,
          "bytes_out": int(bytes_out), "bytes_in": int(bytes_in)}
    if detail:
        ev["detail"] = detail[:300]
    p = path or LOG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(ev) + "\n")
    return ev


def tail(n: int = 40, path: Path | None = None) -> list[dict]:
    p = path or LOG_PATH
    try:
        lines = p.read_bytes()[-262144:].decode("utf-8", "replace").splitlines()
    except OSError:
        return []
    out = []
    for ln in lines[-max(1, n):]:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def summarize(n: int = 5000, path: Path | None = None) -> dict:
    """Per-rule traffic rollup over the last n events — the Network tab's
    'some statistics, nothing too detailed': requests, bytes each way,
    denials, when it last talked.  Counting only; nothing here re-reads
    bodies because bodies were never written."""
    per: dict[str, dict] = {}
    denied_total = 0
    for ev in tail(n, path):
        v = ev.get("verdict")
        rule = ev.get("rule") or "-"
        r = per.setdefault(rule, {"rule": rule, "requests": 0, "bytes_in": 0,
                                  "bytes_out": 0, "denied": 0, "auth_rejects": 0,
                                  "last_ts": "", "last_purpose": ""})
        if v == "ALLOWED":
            r["requests"] += 1
            r["bytes_in"] += int(ev.get("bytes_in") or 0)
            r["bytes_out"] += int(ev.get("bytes_out") or 0)
            r["last_ts"] = ev.get("ts", "")
            r["last_purpose"] = ev.get("purpose", "")
        elif v == "DENIED":
            r["denied"] += 1
            denied_total += 1
            r["last_ts"] = ev.get("ts", "")
        elif v == "AUTH_REJECT":
            r["auth_rejects"] += 1
    rows = sorted(per.values(), key=lambda r: r["last_ts"], reverse=True)
    return {"rules": rows, "denied_total": denied_total,
            "window": f"last {n} events"}
