"""
tool_facade.py — a simplified tool surface for the FAST (voice) LM.

The Mac + knowledge hosts advertise ~25 granular tools.  A 9B model picks better from a
short, intent-named menu, so the cascade wraps the host behind this facade BEFORE offering
tools to the fast LM: related primitives collapse into one wrapper (mail, files, news, web,
calendar read), noisy/internal ones are hidden, and instant local tools (kb_search, kb_ask,
calculate, weather) stay exactly as they are.  The big LM / research worker are NOT wrapped —
they get the full granular set for deliberate, multi-step work.

The facade only restricts what is CATALOGUED to the fast LM.  Any underlying tool can still
be CALLED directly (the bridge's own machinery — calendar verify-reads, write gating — keeps
working unchanged), and a wrapper call resolves to its underlying primitive.  For debugging,
it traces what's available vs offered vs hidden (on change), and every wrapper→primitive
lock-on.

Wrapper spec (one per entry in WRAPPERS):
    name, description, parameters (JSON-schema shown to the LM), and a `route`:
      • {"target": "tool"}                         — 1:1, args passed straight through
      • {"by": "action", "targets": {...},         — pick the primitive from an arg value;
         "default": "tool", "strip": ["action"]}      `strip` drops control args before dispatch
    optional "rename": {wrapper_arg: primitive_arg}.
"""

import time
import typing as tp


def _obj(props: dict, required: tp.Optional[list] = None) -> dict:
    s = {"type": "object", "properties": props}
    if required:
        s["required"] = required
    return s


# Default wrappers: the noisy, multi-tool, often-slow groups.  Knowledge/weather/built-ins
# are deliberately NOT here — they're instant and distinct, and stay native (see PASSTHROUGH).
WRAPPERS: list = [
    {
        "name": "mail",
        "description": "Read the user's email — their latest messages, a search, or open one by id.",
        "parameters": _obj({
            "action": {"type": "string", "enum": ["recent", "search", "read"],
                       "description": "recent = newest messages; search = find by query; read = open one"},
            "query": {"type": "string", "description": "search terms (action=search)"},
            "id": {"type": "string", "description": "message id (action=read)"},
            "folder": {"type": "string", "description": "mailbox folder, optional"},
        }, ["action"]),
        "route": {"by": "action",
                  "targets": {"recent": "mail_recent", "search": "mail_search", "read": "mail_read"},
                  "default": "mail_recent", "strip": ["action"]},
    },
    {
        "name": "files",
        "description": "Look at the user's files — search for one, list a folder, or read a file's contents.",
        "parameters": _obj({
            "action": {"type": "string", "enum": ["search", "list", "read"],
                       "description": "search = find by query; list = a folder; read = a file's contents"},
            "query": {"type": "string", "description": "search terms (action=search)"},
            "path": {"type": "string", "description": "file or folder path (action=read/list)"},
        }, ["action"]),
        "route": {"by": "action",
                  "targets": {"search": "file_search", "list": "file_list", "read": "file_read"},
                  "default": "file_search", "strip": ["action"]},
    },
    {
        "name": "news",
        "description": "Current headlines and online chatter (news and social — all unreliable; weigh it as such).",
        "parameters": _obj({
            "source": {"type": "string", "enum": ["headlines", "forum"],
                       "description": "headlines = news outlets; forum = social/message-board chatter"},
            "query": {"type": "string", "description": "a topic to focus on, optional"},
        }),
        "route": {"by": "source",
                  "targets": {"headlines": "news_headlines", "forum": "fourchan_catalog"},
                  "default": "news_headlines", "strip": ["source"]},
    },
    {
        "name": "web",
        "description": "Search the live web, or fetch a specific page. Slow — use for recency or when the local knowledge base has nothing.",
        "parameters": _obj({
            "action": {"type": "string", "enum": ["search", "fetch"],
                       "description": "search = web search by query; fetch = retrieve one URL"},
            "query": {"type": "string", "description": "search terms (action=search)"},
            "url": {"type": "string", "description": "page to fetch (action=fetch)"},
        }, ["action"]),
        "route": {"by": "action",
                  "targets": {"search": "web_search", "fetch": "web_fetch"},
                  "default": "web_search", "strip": ["action"]},
    },
    {
        "name": "calendar_read",
        "description": "Check the schedule — what's on today, or events across a range of days.",
        "parameters": _obj({
            "scope": {"type": "string", "enum": ["today", "range"],
                      "description": "today = just today; range = a window of upcoming days"},
            "days": {"type": "integer", "description": "how many days ahead (scope=range)"},
        }),
        "route": {"by": "scope",
                  "targets": {"today": "calendar_today", "range": "calendar_range"},
                  "default": "calendar_range", "strip": ["scope"]},
    },
]

# Underlying tools shown to the fast LM with their NATIVE schema (instant locals + the
# calendar WRITE tools, which stay granular so the confirm/announce/verify gating still keys
# off their real names).  Anything available but neither wrapped nor here is hidden from the
# fast LM (still callable internally) — e.g. file_index, fourchan_thread, mail_list.
PASSTHROUGH: list = [
    "weather", "kb_search", "kb_ask",
    # runtime brain toggle on the knowledge host ("load the field-geology brain"):
    # the host only exposes list/load/unload — non-destructive and reversible;
    # import/eject stay panel-side, they change the master.
    "kb_brain",
    "calendar_create", "calendar_update", "calendar_delete",
]


def wrapper_targets(w: dict) -> list:
    """Every underlying primitive a wrapper might dispatch to."""
    r = w.get("route", {})
    if "target" in r:
        return [r["target"]]
    out = list(r.get("targets", {}).values())
    if r.get("default"):
        out.append(r["default"])
    return out


def resolve(w: dict, args: dict) -> tp.Tuple[str, dict]:
    """(underlying_tool, underlying_args) for a wrapper call. Pure."""
    r = w.get("route", {})
    a = dict(args or {})
    if "target" in r:
        name = r["target"]
    else:
        name = r.get("targets", {}).get(a.get(r["by"])) or r.get("default")
    for k in r.get("strip", []):
        a.pop(k, None)
    for src, dst in r.get("rename", {}).items():
        if src in a:
            a[dst] = a.pop(src)
    return name, a


class FacadeHost:
    """Wraps an inner ToolHost/MultiHost, presenting the simplified surface to the fast LM
    while passing every actual call through to the real host.  Mirrors the host interface
    (active / catalogue / call / call_raw) so the bridge uses it transparently."""

    def __init__(self, inner, wrappers: tp.Optional[list] = None,
                 passthrough: tp.Optional[list] = None, hide: tp.Optional[list] = None,
                 trace: tp.Optional[tp.Callable[[dict], None]] = None):
        self.inner = inner
        self.wrappers = wrappers if wrappers is not None else WRAPPERS
        self.passthrough = set(passthrough if passthrough is not None else PASSTHROUGH)
        self.hide = set(hide or [])
        self.trace = trace
        self._by_name = {w["name"]: w for w in self.wrappers}
        self._last_sig: tp.Optional[str] = None       # so availability is traced only on change

    @property
    def active(self) -> bool:
        return self.inner.active

    async def catalogue(self) -> list:
        inner_cat = list(await self.inner.catalogue())
        avail = {(t.get("function") or {}).get("name", "") for t in inner_cat}
        out: list = []
        covered: set = set()
        for w in self.wrappers:
            targets = wrapper_targets(w)
            covered.update(targets)
            if any(t in avail for t in targets):        # offer a wrapper only if it can reach a real tool
                out.append({"type": "function", "function": {
                    "name": w["name"], "description": w["description"],
                    "parameters": w.get("parameters", _obj({}))}})
        for t in inner_cat:
            n = (t.get("function") or {}).get("name", "")
            if n in self.passthrough and n not in self.hide:
                out.append(t)
        self._trace_availability(avail, out, covered)
        return out

    def _trace_availability(self, avail: set, offered: list, covered: set) -> None:
        if not self.trace:
            return
        shown = [(t.get("function") or {}).get("name", "") for t in offered]
        hidden = sorted(avail - covered - self.passthrough - self.hide)
        sig = repr((sorted(avail), shown, hidden))
        if sig == self._last_sig:                       # unchanged turn-to-turn → don't spam the feed
            return
        self._last_sig = sig
        self.trace({"ts": time.time(), "kind": "tool_facade",
                    "available": sorted(avail), "offered": shown, "hidden": hidden})

    def _route(self, name: str, args: dict) -> tp.Tuple[str, dict]:
        w = self._by_name.get(name)
        if not w:
            return name, (args or {})                   # not a wrapper → direct/internal call, untouched
        tool, targs = resolve(w, args)
        if self.trace:
            self.trace({"ts": time.time(), "kind": "tool_locked",
                        "wrapper": name, "tool": tool, "args": targs})
        return tool, targs

    async def call(self, name: str, arguments: dict) -> str:
        tool, args = self._route(name, arguments)
        return await self.inner.call(tool, args)

    async def call_raw(self, name: str, arguments: dict) -> dict:
        tool, args = self._route(name, arguments)
        return await self.inner.call_raw(tool, args)
