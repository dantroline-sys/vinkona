"""
Tool-host client — Tier-2 "run and fetch" tools (calendar, files, mail, …).

The tools live where the data lives: a small HTTP "tool host" on the Mac Mini
(see MAC_TOOLS.md for the contract; back it with MCP servers or write handlers
directly).  Vinkona is the client — it fetches the tool catalogue, hands it to the
fast LM as OpenAI `tools`, and forwards any tool call the model makes.

Contract (Vinkona ⟶ tool host):
  GET  {url}/tools  -> {"tools": [ {name, description, parameters(JSONSchema)} ]}
  POST {url}/call   <- {"name": str, "arguments": obj}
                    -> {"ok": true, "result": str} | {"ok": false, "error": str}

Everything degrades gracefully: if tools are disabled or the host is unreachable,
catalogue() returns [] and the conversation proceeds tool-free.
"""

import typing as tp


class ToolHost:
    def __init__(self, cfg_tools: dict):
        self.enabled = bool(cfg_tools.get("enabled", False))
        self.url = (cfg_tools.get("url") or "").rstrip("/")
        self.timeout = cfg_tools.get("timeout_s", 20)
        # Optional bearer token (the Mac host relies on the tunnel and sets none; the music
        # host may set auth_token).  Sent as Authorization: Bearer on every request.
        self._auth = cfg_tools.get("auth_token")
        self._catalog: tp.Optional[list] = None     # cached per connection

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.url)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._auth}"} if self._auth else {}

    async def catalogue(self) -> list:
        """OpenAI-style tools array for /v1/chat/completions, or [] if unavailable."""
        if not self.active:
            return []
        if self._catalog is not None:
            return self._catalog
        import aiohttp
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{self.url}/tools", headers=self._headers(),
                                 timeout=aiohttp.ClientTimeout(total=self.timeout)) as r:
                    if r.status != 200:
                        return []
                    data = await r.json()
        except Exception:
            return []
        tools = [{"type": "function", "function": t} for t in (data.get("tools") or [])
                 if t.get("name")]
        self._catalog = tools
        return tools

    async def call_raw(self, name: str, arguments: dict) -> dict:
        """Run one tool and return the STRUCTURED outcome: {ok, result, error}.

        Anything that isn't an explicit `ok: true` from the host — a non-200, a
        timeout/transport failure, a malformed body, or `ok: false` — is ok=False with a
        reason.  Callers must check `ok`; never treat a failed/timed-out call as success
        (e.g. a calendar write that never landed)."""
        if not self.active:
            return {"ok": False, "result": "", "error": "tools are not available"}
        import aiohttp
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(f"{self.url}/call", headers=self._headers(),
                                  json={"name": name, "arguments": arguments},
                                  timeout=aiohttp.ClientTimeout(total=self.timeout)) as r:
                    if r.status != 200:
                        return {"ok": False, "result": "", "error": f"tool host error {r.status}"}
                    data = await r.json()
        except Exception as e:
            return {"ok": False, "result": "", "error": f"tool call failed: {e}"}
        if not isinstance(data, dict):
            return {"ok": False, "result": "", "error": "malformed tool response"}
        if data.get("ok"):
            return {"ok": True, "result": str(data.get("result", "")), "error": ""}
        return {"ok": False, "result": "", "error": str(data.get("error", "unknown"))}

    async def call(self, name: str, arguments: dict) -> str:
        """Run one tool; always returns a string for the model (errors included)."""
        d = await self.call_raw(name, arguments)
        return d["result"] if d["ok"] else f"(tool error: {d['error']})"


class MultiHost:
    """Aggregate several ToolHosts behind the single interface the bridge expects, so the
    fast LM can be offered (and can call) tools from more than one host — e.g. the Mac tool
    host plus a separate music host.  catalogue() unions them (first host wins a name
    clash); call/call_raw dispatch to whichever host advertises the tool."""

    def __init__(self, hosts: list):
        self.hosts = [h for h in hosts if h is not None]
        self._owner: dict = {}                       # tool name -> owning host

    @property
    def active(self) -> bool:
        return any(h.active for h in self.hosts)

    async def catalogue(self) -> list:
        cat, seen = [], set()
        self._owner = {}
        for h in self.hosts:
            if not h.active:
                continue
            for t in await h.catalogue():
                name = (t.get("function") or {}).get("name")
                if name and name not in seen:
                    seen.add(name)
                    cat.append(t)
                    self._owner[name] = h
        return cat

    def _host_for(self, name: str):
        # The bridge catalogues before every turn, so _owner is fresh; fall back to the
        # first active host if a call somehow precedes a catalogue.
        return self._owner.get(name) or next((h for h in self.hosts if h.active), None)

    async def call_raw(self, name: str, arguments: dict) -> dict:
        h = self._host_for(name)
        if h is None:
            return {"ok": False, "result": "", "error": "tools are not available"}
        return await h.call_raw(name, arguments)

    async def call(self, name: str, arguments: dict) -> str:
        h = self._host_for(name)
        if h is None:
            return f"(no tool named {name} is available)"
        return await h.call(name, arguments)
