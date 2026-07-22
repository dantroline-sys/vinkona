"""The user's window onto the broker (B-11):

    python3 -m knowledgehost.amiga_net.status [--n 40]

Plain language, three sections: what this machine is ALLOWED to reach and why;
any lease open right now; and the last N things that actually happened.  If
this output surprises you, that is the feature working.
"""
from __future__ import annotations

import argparse

from . import audit, policy


def render(n: int = 40) -> str:
    rules = policy.load()
    out = ["Outbound network policy (egress.toml) — deny by default:", ""]
    if not rules:
        out.append("  (no rules — this machine's code can reach NOTHING outside)")
    for r in rules:
        kind = (f"lease: opens per operation, self-closes after "
                f"{int(r.ttl_seconds)}s" if r.leased else
                "standing: allowed whenever the code asks")
        out.append(f"  • {r.name}: {', '.join(r.hosts)} :{r.port} "
                   f"[{'/'.join(r.methods)}]")
        out.append(f"      why: {r.purpose or '(no purpose given!)'}")
        out.append(f"      {kind}")
    out.append("")
    live = policy.live_leases(rules)
    if live:
        out.append("Open leases right now:")
        for d in live:
            out.append(f"  • {d['rule']} — {d.get('purpose', '')} "
                       f"({d['remaining_s']}s remaining, {d.get('uses', 0)} request(s))")
    else:
        out.append("Open leases right now: none — nothing may leave this "
                   "machine until an operation opens one (standing rules excepted).")
    out.append("")
    events = audit.tail(n)
    out.append(f"Last {len(events)} event(s) (var/log/egress.jsonl):")
    if not events:
        out.append("  (none logged yet)")
    for e in events:
        dest = f"{e.get('host')}:{e.get('port')}" if e.get("host") else e.get("rule", "-")
        size = ""
        if e.get("bytes_in"):
            size = f"  {e['bytes_in'] / 2**20:.1f} MB in"
        out.append(f"  {e.get('ts', '')}  {e.get('verdict', ''):<11} {dest:<34} "
                   f"{e.get('purpose', '')}{size}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="audit events to show")
    args = ap.parse_args()
    print(render(args.n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
