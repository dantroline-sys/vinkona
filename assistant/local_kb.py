"""
LocalKB — Vinkona's in-process knowledge tier (VINUR-PACK-01 §1.6).

The vinur read path consumed as a LIBRARY: no knowledge-host process, no
distill/ingest machinery — just imported knowledge packs (.kdb) in a local
master kb, queried in-process.  Vinur's `knowledgehost` package has zero hard
dependencies by ratchet, so importing it drags in nothing; it is found from
`local_kb.vinur_path`, an installed package, or the paired ../vinur checkout
(the same sibling convention as vinur's control_dir default).

Duck-types knowledge_host.KnowledgeHost — `enabled`, `ask()`, `search()`,
`swap_in()` with the same signatures and the same result shapes (it routes
through vinur's own Tools.call, the exact code the HTTP host serves, and does
the client's double-JSON unwrap) — so the cascade cannot tell the tiers apart.

Fail-soft like the client: construction failure leaves `.enabled` False with
the reason on `.reason`; ask/search return None on any trouble, never raise.
Without an embedder endpoint the read path serves its lexical tier (dense
retrieval and foreign-pack re-embedding need `local_kb.embed_url`).

CLI (smoke/admin):  python3 assistant/local_kb.py status|import <pack>|ask "q"
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("local_kb")

HERE = Path(__file__).resolve().parent          # …/vinkona/assistant


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (HERE / p)


def _find_vinur(explicit: str = ""):
    """Path to insert on sys.path so `knowledgehost` imports, or None when it
    already imports (installed package).  Raises with every location tried."""
    if explicit:
        root = Path(explicit).expanduser()
        if (root / "knowledgehost").is_dir():
            return root
        raise RuntimeError(f"local_kb.vinur_path={explicit!r} has no knowledgehost/ package")
    try:
        import knowledgehost  # noqa: F401 — installed (or already on sys.path)
        return None
    except ImportError:
        pass
    sibling = HERE.parent.parent / "vinur"      # …/vinkona/../vinur — the paired checkout
    if (sibling / "knowledgehost").is_dir():
        return sibling
    raise RuntimeError(
        "vinur's knowledgehost package not found (tried: installed package, "
        f"{sibling}) — set local_kb.vinur_path to a vinur checkout")


class LocalKB:
    """In-process stand-in for the KnowledgeHost client, over imported packs."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.url = ""                            # parity field: there is no host
        self.reason = ""
        self._tools = None
        self._kb = None
        self._store = None
        self._embedder = None
        self._mods = None
        self._cfg = None
        try:
            self._init(cfg)
        except Exception as e:                   # fail-soft: no local tier, never a crash
            self.reason = f"{type(e).__name__}: {e}"
            log.warning("local kb unavailable: %s", self.reason)

    def _init(self, cfg: dict):
        root = _find_vinur(str(cfg.get("vinur_path") or ""))
        if root is not None and str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from knowledgehost import bundles as kh_bundles
        from knowledgehost import config as kh_config
        from knowledgehost import embed as kh_embed
        from knowledgehost import kb as kh_kb
        from knowledgehost import store as kh_store
        from knowledgehost import tools as kh_tools
        base = dict(kh_config.DEFAULTS)          # vinur's own defaults fill every knob
        base.update({
            "kb_path": str(_resolve(str(cfg.get("kb_path") or "var/kb/kb.db"))),
            "db_path": str(_resolve(str(cfg.get("index_path") or "var/kb/index.db"))),
            # empty embed_url = the deliberate lexical-only setup (Embedder would
            # otherwise poll vinur's default localhost endpoint that isn't there)
            "embed_url": str(cfg.get("embed_url") or ""),
        })
        if cfg.get("embed_model"):
            base["embed_model"] = str(cfg["embed_model"])
        self._kb = kh_kb.KB(base)
        self._store = kh_store.make_store(base)
        self._embedder = kh_embed.Embedder(base) if base["embed_url"] else None
        self._tools = kh_tools.Tools(self._store, self._embedder, base, kb=self._kb)
        self._mods = {"kb": kh_kb, "tools": kh_tools, "bundles": kh_bundles}
        self._cfg = base

    # ── the KnowledgeHost surface ────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        return self._tools is not None

    def _call(self, name: str, args: dict):
        """Tools.call + the same double-JSON unwrap the HTTP client performs."""
        if self._tools is None:
            return None
        try:
            outer = self._tools.call(name, args)
        except Exception:                        # Tools.call already shields; belt+braces
            log.exception("local kb tool %s failed", name)
            return None
        if not isinstance(outer, dict) or not outer.get("ok"):
            return None
        result = outer.get("result")
        if isinstance(result, str):
            try:
                return json.loads(result)
            except (ValueError, json.JSONDecodeError):
                return None
        return result if isinstance(result, dict) else None

    async def ask(self, query: str, *, rigor: str = "low",
                  context_features=None, intent: str = "", http=None):
        """kb_ask in-process (same args/result shape as the HTTP client)."""
        query = (query or "").strip()
        if not query:
            return None
        args: dict = {"query": query}
        if rigor and rigor != "low":
            args["rigor"] = rigor
        if context_features:
            args["context_features"] = context_features
        if intent:
            args["intent"] = intent
        return await asyncio.to_thread(self._call, "kb_ask", args)

    async def search(self, query: str, *, intent: str = "", k: int = 5, http=None):
        """kb_search in-process (lexical tier without an embedder)."""
        query = (query or "").strip()
        if not query:
            return None
        args: dict = {"query": query, "k": int(k)}
        if intent:
            args["intent"] = intent
        return await asyncio.to_thread(self._call, "kb_search", args)

    async def swap_in(self, name: str, http=None):
        return None                              # nothing to swap in-process

    # ── pack administration (CLI / future setup surface) ────────────────────
    def import_pack(self, path: str, *, name: str | None = None, trust: str = "low") -> dict:
        """Absorb a .kdb pack into the local master (content-hash idempotent,
        trust-capped; the very first import bootstraps an empty master)."""
        if self._mods is None:
            raise RuntimeError(f"local kb unavailable: {self.reason}")
        res = self._mods["bundles"].import_bundle(self._cfg, str(path),
                                                  name=name, trust=trust)
        self._reopen()                           # drop lazy caches so new rows serve
        return res

    def _reopen(self):
        try:
            self._kb.close()
        except Exception:
            pass
        self._kb = self._mods["kb"].KB(self._cfg)
        self._tools = self._mods["tools"].Tools(self._store, self._embedder,
                                                self._cfg, kb=self._kb)

    def counts(self) -> dict:
        return self._kb.counts() if self._kb is not None else {}

    def close(self):
        for obj in (self._kb, self._store):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass
        self._tools = None


def choose_backend(cfg: dict, kh_mod, localkb_cls=None):
    """The knowledge tier ladder, in one place: remote host > local library >
    None (the Wikipedia built-in then steps up on its own `tools.wikipedia`
    logic).  Returns an object duck-typing KnowledgeHost, or None."""
    kc = cfg.get("knowledge_host") or {}
    if kc.get("enabled"):
        return kh_mod.KnowledgeHost(kc.get("url", ""), token=kc.get("token", ""),
                                    timeout_s=kc.get("timeout_s", 4.0))
    lc = cfg.get("local_kb") or {}
    if lc.get("enabled"):
        lk = (localkb_cls or LocalKB)(lc)
        if lk.enabled:
            return lk
        log.warning("local_kb enabled but unavailable: %s", lk.reason)
    return None


# ── tiny admin CLI ────────────────────────────────────────────────────────────
def _main(argv: list) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.path.insert(0, str(HERE))
    import config as vk_config
    lc = (vk_config.load_config() or {}).get("local_kb") or {}
    lk = LocalKB(lc)
    verb = argv[0] if argv else "status"
    if verb == "status":
        if not lk.enabled:
            print(f"local kb UNAVAILABLE: {lk.reason}")
            return 1
        print(f"local kb at {lk._cfg['kb_path']}: {lk.counts()}")
        return 0
    if verb == "import" and len(argv) >= 2:
        if not lk.enabled:
            print(f"local kb UNAVAILABLE: {lk.reason}")
            return 1
        res = lk.import_pack(argv[1], trust=("keep" if "--keep-trust" in argv else "low"))
        print(json.dumps(res, indent=2))
        return 0
    if verb == "ask" and len(argv) >= 2:
        res = asyncio.run(lk.ask(argv[1]))
        print(json.dumps(res, indent=2, ensure_ascii=False) if res else "no answer")
        return 0
    print("usage: local_kb.py status | import <pack.kdb> [--keep-trust] | ask \"<query>\"")
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
