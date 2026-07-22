"""amiga_net — Vinkona's egress broker (AMIGA-OPS-01 §3.3).

The single place in this codebase that opens an outbound network connection.
Everything else talks to loopback peers (the cascade, the llama-servers, the
config server) or calls this package:

    from .amiga_net import broker
    data = broker.request("fetch the llama-server release", url)   # small sync
    broker.download("llama-server binary", url, dest, sha256=…)    # big files
    async with broker.session("research: arxiv", "research") as s: # async lane
        async with s.get(url, params=…) as r: ...

Deny-by-default: a request is allowed only when it matches a rule in
egress.toml (assistant/egress.toml) — name, host patterns, port, methods, a
plain-language purpose.  A LEASE rule (ttl_seconds/max_uses) grants nothing
until an operation opens it and closes itself, so an idle Vinkona has zero
standing egress.  Every decision is appended to var/log/egress.jsonl:
timestamp, component (vinkona), purpose, destination, rule, verdict, bytes —
never bodies, never tokens.

    python3 -m assistant.amiga_net.status         # the user's window

SAME policy-file and audit-line formats as Vinur's broker (B-13): the future
native snitch daemon reads both.  The async lane exists because Vinkona's
research/wikipedia egress is aiohttp — BrokerSession is the ONE permitted
ClientSession (B-12).
"""
from . import broker  # noqa: F401  (the public surface)
