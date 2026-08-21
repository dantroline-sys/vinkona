"""
local_tools — the Mac tool host's genres, bundled into Vinkona (VIN-LOCAL-01).

The same catalogue MAC_TOOLS.md specifies of any tool host — files, calendar,
mail, news, weather, and the keyless research tools — served IN-PROCESS on the
box Vinkona runs on, for setups with no Mac (or no wish to keep one awake).
LocalHost duck-types tools_client.ToolHost (active / catalogue / call_raw /
call), so it simply joins the MultiHost list; every consumer — the fast LM's
tool loop, the mail/file crawls, calendar sync, news ingestion, research
routing — already speaks call_raw and works unchanged.  A real Mac host
outranks it on a name clash (MultiHost lists it first), so the richer
implementation (Spotlight, OCR) wins wherever both are enabled.

Ground rules:
  • Per-genre opt-in, everything OFF by default.  A genre that is enabled but
    unconfigured (no roots, no feeds, no account) answers honestly that it
    needs configuring rather than pretending to be empty.
  • Read-only, except calendar writes — and those go ONLY to the calendar
    named in tools.local.calendar.vinkona_calendar, enforced HERE, exactly as
    MAC_TOOLS.md demands of any host.  There is deliberately no mail_send.
  • All HTTP egress goes through the amiga_net broker against egress.toml;
    egress_sync.py keeps a managed rule block derived from what the user
    actually configured, so deny-by-default stays true.  IMAP is the one
    non-HTTP lane: direct TLS from imaplib to the server the user configured,
    called out in the managed block so the posture file stays honest.
  • Handlers are synchronous and injectable (fetch / imap_factory / now), and
    run under asyncio.to_thread with a cap — a slow server can spoil one
    call, never the loop.  Tests run every genre offline on canned transports.
"""
import asyncio
import importlib
import time
import typing as tp

# Genre → module (calendar's module is caldav.py: the stdlib calendar module
# owns that name).  Modules import lazily so an unused genre costs nothing.
GENRE_MODULES = {"files": "files", "news": "news", "weather": "weather",
                 "research": "research", "mail": "mail", "calendar": "caldav"}
GENRES = tuple(GENRE_MODULES)


def enabled_genres(lcfg: dict) -> list:
    return [g for g in GENRES if (lcfg.get(g) or {}).get("enabled")]


def broker_fetch(url: str, *, method: str = "GET", data: bytes | None = None,
                 headers: dict | None = None, timeout: float = 15.0,
                 purpose: str = "local_tools") -> tuple:
    """(status, body) via the amiga_net broker — every call policy-checked and
    audited against egress.toml.  An HTTP error status comes back as data (the
    genre decides what it means); an egress denial raises, so the user sees the
    same actionable 'add a rule deliberately' message everywhere."""
    import urllib.error
    from amiga_net import broker
    try:
        body = broker.request(purpose, url, method=method, data=data,
                              headers=headers, timeout=timeout)
        return 200, body
    except urllib.error.HTTPError as e:
        try:
            return int(e.code), e.read() or b""
        except Exception:
            return int(e.code), b""


class LocalHost:
    """The bundled toolset behind the ToolHost interface.

    `cfg` is the WHOLE config document (genres read their own block under
    tools.local; calendar also reads the user's timezone from the top-level
    calendar block).  `env` overrides inject transports for tests:
    fetch / imap_factory / now / news_db_path / uuid.
    """

    def __init__(self, cfg: dict, *, news_db_path: str = "",
                 env: dict | None = None, trace=None):
        self.cfg = cfg or {}
        self.lcfg = (self.cfg.get("tools") or {}).get("local") or {}
        self.timeout = float(self.lcfg.get("timeout_s", 20))
        self.trace = trace
        self._env = dict(env or {})
        self._env.setdefault("news_db_path", news_db_path)
        self._handlers: tp.Optional[dict] = None
        self._catalog: tp.Optional[list] = None

    @property
    def active(self) -> bool:
        return bool(self.lcfg.get("enabled")) and bool(enabled_genres(self.lcfg))

    # ── catalogue assembly ────────────────────────────────────────────────
    def _build(self) -> None:
        if self._handlers is not None:
            return
        env = dict(self._env)
        env.setdefault("fetch", broker_fetch)
        env.setdefault("now", time.time)
        env.setdefault("user_tz", (self.cfg.get("tools") or {})
                       .get("calendar", {}).get("timezone", ""))
        handlers, cat = {}, []
        for genre in enabled_genres(self.lcfg):
            try:
                mod = importlib.import_module(f"{__name__}.{GENRE_MODULES[genre]}")
                pairs = mod.tools(self.lcfg.get(genre) or {}, env)
            except Exception:
                continue                    # one broken genre never sinks the rest
            for spec, fn in pairs:
                if spec["name"] in handlers:
                    continue
                handlers[spec["name"]] = fn
                cat.append({"type": "function", "function": spec})
        self._handlers, self._catalog = handlers, cat
        # Keep egress.toml's managed block current with what is actually
        # configured (cheap no-op when nothing changed; never fatal).
        try:
            from . import egress_sync
            egress_sync.ensure(self.cfg)
        except Exception:
            pass

    # ── the ToolHost interface ────────────────────────────────────────────
    async def catalogue(self) -> list:
        if not self.active:
            return []
        self._build()
        return self._catalog

    async def call_raw(self, name: str, arguments: dict) -> dict:
        if not self.active:
            return {"ok": False, "result": "", "error": "local tools are not available"}
        self._build()
        fn = self._handlers.get(name)
        if fn is None:
            return {"ok": False, "result": "", "error": f"no local tool named {name}"}
        try:
            out = await asyncio.wait_for(
                asyncio.to_thread(fn, dict(arguments or {})), timeout=self.timeout)
        except asyncio.TimeoutError:
            return {"ok": False, "result": "",
                    "error": f"{name} timed out after {self.timeout:.0f}s"}
        except Exception as e:
            return {"ok": False, "result": "", "error": f"{name} failed: {e}"}
        if isinstance(out, dict) and "ok" in out:
            return {"ok": bool(out.get("ok")), "result": str(out.get("result", "")),
                    "error": str(out.get("error", ""))}
        return {"ok": True, "result": str(out), "error": ""}

    async def call(self, name: str, arguments: dict) -> str:
        d = await self.call_raw(name, arguments)
        return d["result"] if d["ok"] else f"(tool error: {d['error']})"


def probe(cfg: dict, genre: str, *, news_db_path: str = "",
          env: dict | None = None) -> dict:
    """One bounded, synchronous 'does this genre's configuration actually
    work?' check for the panel/desk Test buttons.  Returns {ok, detail} and
    never raises — the detail is written for the person reading the panel."""
    lcfg = (cfg.get("tools") or {}).get("local") or {}
    gcfg = dict(lcfg.get(genre) or {})
    if genre not in GENRES:
        return {"ok": False, "detail": f"unknown genre {genre!r}"}
    full_env = {"fetch": broker_fetch, "now": time.time, "news_db_path": news_db_path,
                "user_tz": (cfg.get("tools") or {}).get("calendar", {}).get("timezone", ""),
                **(env or {})}
    try:
        mod = importlib.import_module(f"{__name__}.{GENRE_MODULES[genre]}")
        return mod.probe(gcfg, full_env)
    except Exception as e:
        return {"ok": False, "detail": f"{genre} test failed: {e}"}


from . import egress_sync              # noqa: E402  (tiny, dependency-free — eager so
#                                        callers can reach local_tools.egress_sync directly)
