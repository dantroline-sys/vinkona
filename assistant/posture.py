"""Posture: Vinkona's honest "how leaky is this box?" scan (AMIGA-OPS-01
B-18..23).  Read-only and advisory — it reports, it changes nothing.

Three lights, and UNKNOWN is its own state (a check that could not run is never
a pass, B-21): good / warn (works, worth fixing — the fix is named) / bad
(leaking or refusing to work).  Overall = the worst light present.

Vinkona-specific vs Vinur's: the one deliberate non-loopback bind is the
cascade WSS (server.host 0.0.0.0) so the phone can reach it — graded by
whether its first-frame token auth is on.  No vLLM phone-home to verify.
Lives beside amiga_net/ (not inside it), same as Vinur.
"""
from __future__ import annotations

import os
import stat
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent           # the assistant/ app root

# amiga_net loads either way: package context (tests / assistant.posture) or
# the script-tree top-level the app actually runs in.
try:
    from .amiga_net import audit as _audit, policy as _policy
except Exception:                                # pragma: no cover
    from amiga_net import audit as _audit, policy as _policy

_ORDER = {"bad": 0, "unknown": 1, "warn": 2, "good": 3}


def _c(cid, name, state, detail, fix=""):
    out = {"id": cid, "name": name, "state": state, "detail": detail}
    if fix:
        out["fix"] = fix
    return out


# ── pure parsers (testable without a live /proc) ─────────────────────────────

def parse_proc_tcp(text: str, v6: bool = False) -> list:
    out = []
    for ln in text.splitlines()[1:]:
        parts = ln.split()
        if len(parts) < 4 or parts[3] != "0A":        # 0A = LISTEN
            continue
        addr, _, port_hex = parts[1].rpartition(":")
        try:
            port = int(port_hex, 16)
        except ValueError:
            continue
        if set(addr) <= {"0"}:
            kind = "wildcard"
        elif (not v6 and addr.endswith("7F")) or \
                (v6 and addr == "00000000000000000000000001000000"):
            kind = "loopback"
        else:
            kind = "local"
        out.append((kind, port))
    return out


def wg_interfaces(sysroot: Path = Path("/sys/class/net")) -> list | None:
    if not sysroot.is_dir():
        return None
    out = []
    for d in sorted(sysroot.iterdir()):
        try:
            if "DEVTYPE=wireguard" in (d / "uevent").read_text():
                oper = "?"
                try:
                    oper = (d / "operstate").read_text().strip()
                except OSError:
                    pass
                out.append((d.name, oper))
        except OSError:
            continue
    return out


def expected_ports(cfg: dict) -> dict:
    """port -> (service, needs_auth).  needs_auth ports may legitimately be
    LAN-bound (the phone reaches the cascade); loopback-only ports must not."""
    out: dict = {}
    srv = cfg.get("server") or {}
    try:
        out[int(srv.get("port") or 8998)] = ("cascade WSS (phone)", True)
    except (TypeError, ValueError):
        pass
    for key, default, label in (
            ("config_server", 8090, "config server"),
            ("tts", 11436, "tts"),
            ("fast_lm", 11434, "fast LM"),
            ("big_lm", 11438, "big LM"),
            ("embed", 11437, "embed")):
        sec = cfg.get(key) or {}
        p = sec.get("port")
        if p:
            try:
                out[int(p)] = (label, False)
            except (TypeError, ValueError):
                pass
    return out


def cascade_auth_on(cfg: dict) -> bool:
    return bool(((cfg.get("server") or {}).get("auth") or {}).get("require_auth", True))


# ── the checks ───────────────────────────────────────────────────────────────

def check_policy() -> dict:
    policy = _policy
    if not policy.POLICY_PATH.exists():
        return _c("policy", "Egress policy", "bad",
                  "egress.toml is missing — the broker denies everything, so "
                  "research/wikipedia lookups will fail",
                  "restore egress.toml from the repo")
    rules = policy.load()
    if not rules:
        return _c("policy", "Egress policy", "bad",
                  "egress.toml exists but no rule parses — deny-by-default "
                  "holds (nothing leaks) but every lookup will fail",
                  "fix the TOML; `python3 -m assistant.amiga_net.status` shows what loaded")
    standing = [r.name for r in rules if r.enabled and not r.leased]
    off = [r.name for r in rules if not r.enabled]
    if standing:
        return _c("policy", "Egress policy", "warn",
                  f"rule(s) {', '.join(standing)} are STANDING — open at all times",
                  "add ttl_seconds/max_uses to make them leases")
    detail = f"{len(rules)} rule(s), all lease-only — idle Vinkona has zero standing egress"
    if off:
        detail += f"; disabled: {', '.join(off)}"
    return _c("policy", "Egress policy", "good", detail)


def check_audit() -> dict:
    audit = _audit
    d = audit.LOG_PATH.parent
    if not os.access(d if d.is_dir() else d.parent, os.W_OK):
        return _c("audit", "Audit log", "bad",
                  f"{audit.LOG_PATH} is not writable — egress would go unrecorded",
                  "fix permissions on var/log")
    evs = audit.tail(1)
    last = f"last event {evs[0]['ts']}" if evs else "no events yet"
    return _c("audit", "Audit log", "good", f"append-only at {audit.LOG_PATH} ({last})")


def check_listeners(cfg: dict, rows: list | None) -> dict:
    if rows is None:
        return _c("listen", "Listening sockets", "unknown",
                  "cannot read /proc/net/tcp on this OS yet — what binds where "
                  "was NOT verified",
                  "check by hand: ss -tlnp (Linux) / netstat -an")
    known = expected_ports(cfg)
    auth_on = cascade_auth_on(cfg)
    exposed_ungated, exposed_gated, exposed_other = [], [], []
    for kind, port in rows:
        if kind == "loopback":
            continue
        if port in known:
            label, needs_auth = known[port]
            if needs_auth:
                (exposed_gated if auth_on else exposed_ungated).append(f"{label} (:{port})")
            else:
                exposed_ungated.append(f"{label} (:{port}) — should be loopback-only")
        else:
            exposed_other.append(f":{port}")
    if exposed_ungated:
        return _c("listen", "Listening sockets", "bad",
                  f"{', '.join(sorted(set(exposed_ungated)))} reachable from the "
                  "network without auth",
                  "the cascade: set server.auth.require_auth true; other services: "
                  "bind host 127.0.0.1")
    if exposed_gated:
        return _c("listen", "Listening sockets", "warn",
                  f"{', '.join(sorted(set(exposed_gated)))} reachable from the "
                  "network (first-frame token required — a declared deployment for "
                  "the phone, not a leak)"
                  + (f"; not Vinkona's: {', '.join(sorted(set(exposed_other)))}"
                     if exposed_other else ""),
                  "keep it deliberate; a WireGuard overlay is safer off-LAN")
    detail = "the cascade is loopback (no phone deployment) — everything is local"
    if exposed_other:
        detail = (f"every Vinkona port is loopback — but OTHER software listens "
                  f"openly on {', '.join(sorted(set(exposed_other))[:8])}; not "
                  "Vinkona's to fix, worth knowing")
        return _c("listen", "Listening sockets", "warn", detail, "identify them: ss -tlnp")
    return _c("listen", "Listening sockets", "good", detail)


def check_wireguard(wgs: list | None, lan_exposed: bool) -> dict:
    if wgs is None:
        return _c("wg", "WireGuard overlay", "unknown",
                  "cannot inspect network interfaces on this OS yet")
    up = [n for n, oper in wgs if oper in ("up", "unknown")]
    if up:
        return _c("wg", "WireGuard overlay", "good",
                  f"interface {', '.join(up)} is up — the phone/peers ride an "
                  "encrypted overlay")
    if wgs:
        return _c("wg", "WireGuard overlay", "warn",
                  f"configured ({', '.join(n for n, _ in wgs)}) but not up",
                  "wg-quick up <iface>")
    if lan_exposed:
        return _c("wg", "WireGuard overlay", "warn",
                  "no overlay, and the cascade is LAN-reachable — the phone link "
                  "is only as safe as your network",
                  "consider WireGuard between phone and box (B-3's other half)")
    return _c("wg", "WireGuard overlay", "good",
              "not configured — and not needed: the cascade is loopback")


def check_token(cfg: dict) -> dict:
    tok = str(cfg.get("hf_token") or "").strip()
    if not tok:
        return _c("token", "HF token storage", "good",
                  "no token stored — nothing to leak (gated asset repos would ask for one)")
    cp = str(cfg.get("_config_path") or "")
    if cp and Path(cp).exists():
        mode = stat.S_IMODE(Path(cp).stat().st_mode)
        if mode & 0o044:
            return _c("token", "HF token storage", "warn",
                      f"the config file holds the token and is group/world-readable "
                      f"(mode {oct(mode)[2:]})", f"chmod 600 {cp}")
        return _c("token", "HF token storage", "good",
                  "in the config file, owner-readable only; attached by the broker")
    return _c("token", "HF token storage", "good",
              "from the environment; attached by the broker only")


def check_proxy(cfg: dict) -> dict:
    keys = ("http_proxy", "https_proxy", "all_proxy")
    if not any(os.environ.get(k) or os.environ.get(k.upper()) for k in keys) \
            and not any(cfg.get(k) for k in keys):
        return _c("proxy", "Proxy hygiene", "good", "no proxy configured (direct egress)")
    no_p = (os.environ.get("no_proxy") or os.environ.get("NO_PROXY")
            or str(cfg.get("no_proxy") or ""))
    if not ({h.strip() for h in no_p.split(",")} & {"127.0.0.1", "localhost", "::1"}):
        return _c("proxy", "Proxy hygiene", "warn",
                  "a proxy is set but no_proxy doesn't exempt loopback — the box's "
                  "own calls to its llama-servers would be sent to the proxy",
                  "add no_proxy=localhost,127.0.0.1,::1")
    return _c("proxy", "Proxy hygiene", "good", "proxy set, loopback exempt")


def check_installs() -> dict:
    return _c("installs", "Install-time fetches", "warn",
              "install/update scripts (uv, pip, git, llama.cpp fetch, HF asset "
              "downloads) reach the network directly — confinement begins "
              "POST-install by design",
              "run installs deliberately; runtime egress is broker-only")


# ── the scan ─────────────────────────────────────────────────────────────────

def scan(cfg: dict) -> dict:
    try:
        rows = parse_proc_tcp(Path("/proc/net/tcp").read_text())
        rows += parse_proc_tcp(Path("/proc/net/tcp6").read_text(), v6=True)
    except OSError:
        rows = None
    lan = bool(rows) and any(k != "loopback" and p in expected_ports(cfg)
                             for k, p in rows)
    checks = [check_policy(), check_audit(), check_listeners(cfg, rows),
              check_wireguard(wg_interfaces(), lan), check_token(cfg),
              check_proxy(cfg), check_installs()]
    counts = {"good": 0, "warn": 0, "bad": 0, "unknown": 0}
    for c in checks:
        counts[c["state"]] = counts.get(c["state"], 0) + 1
    overall = min((c["state"] for c in checks), key=lambda s: _ORDER[s])
    return {"checks": checks, "summary": {**counts, "overall": overall,
                                          "at": time.time()}}
