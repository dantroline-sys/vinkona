"""The Network tab's server side: the egress broker's view and controls.

Read: posture (leak check), rules + live leases, per-rule traffic stats, the
audit tail, and the broker settings (proxy redacted, token write-only).
Write: revoke a lease, enable/disable a rule, save one setting.  Mirrors
Vinur's /net endpoint so the two panels feel the same.
"""
from __future__ import annotations

import re
import shutil

import posture                                    # script-tree top-level modules
from amiga_net import audit, broker, policy

NET_KEYS = ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
            "fetch_engine", "hf_token")


def redact_url(v: str) -> str:
    return re.sub(r"(://)[^/@\s]+:[^/@\s]+@", r"\1***:***@", v or "")


def settings_view(cfg: dict) -> dict:
    out: dict = {}
    for k in NET_KEYS:
        v = str(cfg.get(k) or "")
        if k == "hf_token":
            out["hf_token_set"] = bool(v)
            out["hf_token_hint"] = ("…" + v[-4:]) if len(v) >= 12 else ""
        else:
            out[k] = redact_url(v)
    return out


def view(cfg: dict) -> dict:
    """Everything the Network tab shows."""
    rules = policy.load()
    live = {d["rule"]: d for d in policy.live_leases(rules)}
    rule_rows = [{"name": r.name, "purpose": r.purpose, "hosts": r.hosts,
                  "port": r.port, "methods": r.methods, "leased": r.leased,
                  "enabled": r.enabled, "auth": bool(r.auth),
                  "lease": live.get(r.name)} for r in rules]
    try:
        pos = posture.scan(cfg)
    except Exception as e:                            # a broken check must not
        pos = {"checks": [], "summary": {"overall": "unknown",  # take the tab down
                                         "error": f"{type(e).__name__}: {e}"}}
    return {"ok": True, "settings": settings_view(cfg),
            "engines": {"aria2c": bool(shutil.which("aria2c")),
                        "wget": bool(shutil.which("wget"))},
            "engine_resolved": broker._engine(),
            "rules": rule_rows, "stats": audit.summarize(),
            "events": audit.tail(10), "audit_path": str(audit.LOG_PATH),
            "posture": pos}


def action(req: dict) -> dict:
    """A Network-tab action: revoke a lease or toggle a rule.  Audited."""
    act = str(req.get("action") or "")
    rule = str(req.get("rule") or "")
    if act == "revoke_lease":
        policy.lease_close(rule)
        audit.write("POLICY", rule=rule or "-",
                    detail="lease revoked by operator (Network tab)")
        return {"ok": True, "note": f"lease on '{rule}' revoked — whatever holds "
                "it is refused on its next request"}
    if act == "rule":
        on = bool(req.get("enabled"))
        try:
            policy.set_rule_enabled(rule, on)
        except (ValueError, OSError) as e:
            return {"ok": False, "error": str(e)}
        if not on:
            policy.lease_close(rule)
        audit.write("POLICY", rule=rule,
                    detail=("rule enabled" if on else "rule disabled")
                           + " by operator (Network tab)")
        return {"ok": True, "note": f"rule '{rule}' "
                + ("enabled" if on else "disabled — nothing can use or lease it")}
    return {"ok": False, "error": f"unknown action {act}"}


def set_setting(config_path, key: str, value) -> dict:
    """Save one broker/network setting into config.json (whole-file atomic
    write, the way this UI already saves config).  The redacted echo is
    refused so it can never clobber a real proxy credential."""
    import json
    import os
    from pathlib import Path
    if key not in NET_KEYS:
        return {"ok": False, "error": f"not a network setting: {key}"}
    v = str(value if value is not None else "").strip()
    if "***" in v:
        return {"ok": False, "error": "that is the REDACTED display form — retype "
                "the real value"}
    if key == "fetch_engine" and v not in ("", "aria2c", "wget", "stdlib"):
        return {"ok": False, "error": "fetch_engine must be aria2c, wget, stdlib, "
                "or empty (auto)"}
    if key in ("http_proxy", "https_proxy", "all_proxy") and v and "://" not in v:
        return {"ok": False, "error": f"{key} should be a URL (http://host:3128)"}
    # whole-file atomic write, the way this UI already saves config.json —
    # only the one key changes, every other setting is preserved
    p = Path(config_path)
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur[key] = v
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(cur, indent=2) + "\n")
    os.replace(tmp, p)
    note = ("held by the broker; engines never see it" if key == "hf_token"
            else "applies to the next research turn / fetch")
    return {"ok": True, "key": key, "note": note}
