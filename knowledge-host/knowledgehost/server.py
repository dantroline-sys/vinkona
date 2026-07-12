"""Stdlib HTTP server exposing the tool contract — light, fast, always up.

Endpoints:
- ``GET  /health``   liveness + index stats (chunk count, backend, dense?)
- ``GET  /tools``    the tool catalogue
- ``POST /call``     run a tool  {name, arguments}  ->  {ok, result|error}

Read-only and **localhost-bound** (it is never on the LAN).  If ``auth_token``
is set, ``/call`` requires ``Authorization: Bearer <token>`` — but on the GPU
box it's co-located with the cascade and needs no tunnel, so a token is
optional.  Threaded so concurrent tool calls don't queue.
"""
from __future__ import annotations

import hmac
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# Cap on a POST body we will read into memory.  The control routes only ever carry
# small JSON; refusing anything larger stops a local process from OOMing the server
# with a giant Content-Length (the server is localhost-only, but cheap to bound).
_MAX_BODY = 4 * 1024 * 1024

from . import __version__
from . import lm_lease
from .ops import COMMANDS as OPS_COMMANDS, OpsRunner
from .viewer import INDEX_HTML

log = logging.getLogger("knowledgehost.server")


def _import_formats() -> list:
    """What can THIS install ingest right now?  Live probes (find_spec/which),
    never assumptions — the viewer's Import table renders exactly this."""
    import importlib.util as _iu
    import shutil as _sh

    def has(mod):
        try:
            return _iu.find_spec(mod) is not None
        except Exception:
            return False

    traf = has("trafilatura")
    ocr = bool(_sh.which("tesseract")) and bool(_sh.which("ocrmypdf"))
    return [
        {"format": "Text / Markdown", "matches": ".txt  .md", "ready": True,
         "how": "always available (stdlib) — Vinkona's research drops are .md"},
        {"format": "HTML", "matches": ".html  .htm", "ready": True,
         "how": ("trafilatura installed — full boilerplate stripping" if traf else
                 "works via the stdlib fallback; ./install.sh --html upgrades extraction")},
        {"format": "PDF", "matches": ".pdf", "ready": has("fitz"),
         "how": "./install.sh --pdf   (PyMuPDF)"},
        {"format": "Scanned-PDF OCR", "matches": "(fallback inside PDF ingest)", "ready": ocr,
         "how": "system packages: tesseract + ocrmypdf (the --pdf install offers them)"},
        {"format": "EPUB", "matches": ".epub", "ready": has("ebooklib"),
         "how": "./install.sh --epub   (ebooklib)"},
        {"format": "Wikipedia ZIM", "matches": ".zim — run: ingest --wikipedia",
         "ready": has("libzim"),
         "how": "./install.sh --wikipedia   (libzim; set zim_path in Settings)"},
    ]


def _external_datasets(cfg: dict) -> list:
    """The bulk-importable external datasets: each with its ops verb, the config
    key holding its file path, and a LIVE does-the-file-exist probe — so the Ops
    tab can show what's ready to import versus what still needs downloading."""
    from pathlib import Path
    sets = [
        ("ConceptNet 5.7", "import-conceptnet", "conceptnet_path",
         "commonsense triples, assertions.csv (~10 GB) — regime=conventional, low trust"),
        ("ATOMIC v4", "import-atomic", "atomic_path",
         "social if-then commonsense, v4_atomic_all_agg.csv — same epistemics as ConceptNet"),
        ("GLUCOSE", "import-glucose", "glucose_path",
         "general causal rules (variable-slot), training CSV — commonsense backbone"),
        ("CauseNet-precision", "import-causenet", "causenet_path",
         "grounded cause→effect graph (JSONL) — has_reference=1, corroboration counts"),
    ]
    out = []
    for name, verb, key, note in sets:
        p = str(cfg.get(key) or "").strip()
        present = bool(p) and Path(p).expanduser().exists()
        out.append({"name": name, "verb": verb, "config_key": key,
                    "path": p or None, "present": present, "note": note})
    return out


def _help_payload(cfg: dict) -> dict:
    """Tab help (help.json, read per request so edits show on refresh) + the
    live import-format probes and external-dataset probes above."""
    from pathlib import Path
    try:
        tabs = json.loads((Path(__file__).parent / "help.json").read_text())
    except Exception:
        tabs = {}
    return {"help": tabs, "formats": _import_formats(),
            "datasets": _external_datasets(cfg)}


class Handler(BaseHTTPRequestHandler):
    server_version = f"knowledgehost/{__version__}"
    protocol_version = "HTTP/1.1"

    @property
    def cfg(self):
        return self.server.cfg

    def log_message(self, fmt, *a):
        log.info("%s - %s", self.address_string(), fmt % a)

    def _authed(self) -> bool:
        token = self.cfg.get("auth_token")
        if not token:
            return True
        got = self.headers.get("Authorization", "").strip()
        # constant-time compare so a wrong token can't be recovered by timing.
        return hmac.compare_digest(got, f"Bearer {token}")

    def _send(self, body: bytes, status=200, ctype="application/json; charset=utf-8"):
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # client navigated away / closed the tab before we finished writing — benign,
            # and not worth a stack trace (the viewer polls, so this happens routinely).
            pass

    def _send_json(self, obj, status=200):
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"), status)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            return None
        if length < 0 or length > _MAX_BODY:            # refuse an absurd body
            return None
        try:
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw or b"{}")
        except (ValueError, json.JSONDecodeError):
            return None

    def _health(self) -> dict:
        """Cheap status for the panel: lease state (is Vinkona on the GPU?) + KB counts."""
        kb = getattr(self.server, "kb", None)
        h = {"counts": kb.counts() if kb else {}}
        try:
            h["lease_fast"] = lm_lease.is_held(lm_lease.FAST, self.cfg)
            h["lease_big"] = lm_lease.is_held(lm_lease.BIG, self.cfg)
        except Exception:
            pass
        return h

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        q = parse_qs(parsed.query)
        store = self.server.store
        if path == "/":                            # the human-facing viewer
            return self._send(INDEX_HTML.encode("utf-8"), ctype="text/html; charset=utf-8")
        if path == "/health":
            return self._send_json({
                "ok": True, "version": __version__,
                "backend": store.backend, "chunks": store.count(),
                "dense": store.has_vectors(),
                "auth_required": bool(self.cfg.get("auth_token"))})
        if path == "/stats":                       # viewer: index breakdown
            by_source = store.stats_by_source() if hasattr(store, "stats_by_source") else {}
            return self._send_json({
                "ok": True, "backend": store.backend, "chunks": store.count(),
                "dense": store.has_vectors(),
                "version": store.manifest.meta_get("version", "1"),
                "by_source": by_source})
        if path == "/sample":                      # viewer: eyeball stored chunks
            n = min(int((q.get("n") or ["20"])[0] or 20), 100)
            src = (q.get("source_type") or [None])[0] or None
            rows = store.sample(n, src) if hasattr(store, "sample") else []
            return self._send_json({"ok": True, "passages": rows})
        if path == "/kb":                          # viewer: structured-KB counts
            kb = getattr(self.server, "kb", None)
            return self._send_json({"ok": True, "counts": kb.counts() if kb else {}})
        if path == "/knowledge":                   # viewer: distilled nodes (the 'learnings')
            kb = getattr(self.server, "kb", None)
            n = min(int((q.get("n") or ["20"])[0] or 20), 100)
            rows = kb.sample_nodes(n) if kb else []
            return self._send_json({"ok": True, "nodes": rows})
        if path == "/facets":                      # viewer: facet coverage per axis (facets.py)
            from .facets import AXES
            kb = getattr(self.server, "kb", None)
            return self._send_json({"ok": True, "axes": list(AXES),
                                    "counts": kb.facet_counts() if kb else {}})
        if path == "/browse":                      # viewer: peruse any produced table
            kb = getattr(self.server, "kb", None)
            kind = (q.get("kind") or ["nodes"])[0]
            n = min(int((q.get("n") or ["50"])[0] or 50), 200)
            fn = {
                "nodes": lambda: kb.sample_nodes(n),
                "edges": lambda: kb.list_edges(n),
                "cards": lambda: kb.list_cards(n),
                "sources": lambda: kb.list_sources(n),
                "adjudication": lambda: kb.list_merge_candidates(n),
                "gaps": lambda: kb.list_gaps(n),
            }.get(kind)
            rows = fn() if (kb and fn) else []
            return self._send_json({"ok": True, "kind": kind, "rows": rows})
        if path == "/search":                      # viewer: run kb_search (no auth, read-only)
            query = (q.get("q") or [""])[0]
            k = int((q.get("k") or ["8"])[0] or 8)
            res = self.server.tools.call("kb_search", {"query": query, "k": k})
            if res.get("ok"):
                return self._send_json({"ok": True, **json.loads(res["result"])})
            return self._send_json(res)
        if path == "/ask":                         # viewer: structured grounded answer
            args = {"query": (q.get("q") or [""])[0]}
            if q.get("rigor"):
                args["rigor"] = q["rigor"][0]
            if q.get("mode"):
                args["mode"] = q["mode"][0]
            if q.get("strict"):
                args["strict"] = q["strict"][0].lower() in ("1", "true", "yes", "on")
            res = self.server.tools.call("kb_ask", args)
            if res.get("ok"):
                return self._send_json({"ok": True, **json.loads(res["result"])})
            return self._send_json(res)
        if path == "/library":                     # viewer/curl: search the document library
            args = {"query": (q.get("q") or [""])[0],
                    "k": int((q.get("k") or ["8"])[0] or 8)}
            if q.get("collection"):
                args["collection"] = q["collection"][0]
            res = self.server.tools.call("library_search", args)
            if res.get("ok"):
                return self._send_json({"ok": True, **json.loads(res["result"])})
            return self._send_json(res)
        if path == "/help":                        # viewer: tab help + import/dataset probes
            return self._send(json.dumps(_help_payload(self.cfg)).encode())
        if path == "/tools":
            return self._send_json(self.server.tools.catalogue())
        # ── control panel (auth-gated when a token is set) ──
        if path == "/bundles":                     # modular §16: groups + scenarios + active
            if not self._authed():
                return self._send_json({"ok": False, "error": "unauthorized"}, 401)
            from . import bundles as B
            kb = getattr(self.server, "kb", None)
            scenarios = self.cfg.get("scenarios") or {}
            return self._send_json({
                "ok": True,
                "bundles": kb.bundle_summary() if kb else [],
                "sources": kb.list_sources(500) if kb else [],
                "scenarios": {n: (scenarios[n] if isinstance(scenarios[n], dict) else {})
                              for n in scenarios},
                "active": B.active_scenario_name(self.cfg),
                "master": self.server.master_kb_path(),
                "working": self.cfg.get("kb_path"),
                "modular": B.is_modular(self.cfg),
                "encrypted_bundles": self.cfg.get("encrypted_bundles") or []})
        if path == "/ops/autopilot":                # Prioritizer tab: the plan + live state
            if not self._authed():
                return self._send_json({"ok": False, "error": "unauthorized"}, 401)
            from . import autopilot as A
            ap = getattr(self.server, "autopilot", None)
            return self._send_json({"ok": True, "plan": A.load_plan(self.cfg),
                                    "state": ap.status() if ap else {"enabled": False},
                                    "commands": OPS_COMMANDS})
        if path in ("/ops/status", "/ops/log", "/config"):
            if not self._authed():
                return self._send_json({"ok": False, "error": "unauthorized"}, 401)
            if path == "/ops/status":
                return self._send_json({"ok": True, "status": self.server.ops.status(),
                                        "health": self._health(),
                                        "commands": OPS_COMMANDS})
            if path == "/ops/log":
                n = int((q.get("tail") or ["300"])[0] or 300)
                return self._send_json({"ok": True, "log": self.server.ops.tail(n),
                                        "status": self.server.ops.status(),
                                        "health": self._health()})
            if path == "/config":
                from .config import settings_schema
                schema = settings_schema()
                return self._send_json({
                    "ok": True, "schema": schema,
                    "values": {k: self.cfg.get(k) for k in schema},
                    "config_path": self.cfg.get("_config_path")})
        if path == "/library/config":               # Library panel: trusted root + subfolder toggles
            if not self._authed():
                return self._send_json({"ok": False, "error": "unauthorized"}, 401)
            from .config import library_status
            return self._send_json({"ok": True, **library_status(self.cfg)})
        return self._send_json({"ok": False, "error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/call", "/ops/run", "/ops/stop", "/ops/reload", "/config",
                        "/ops/autopilot", "/library/config", "/source", "/scenario"):
            return self._send_json({"ok": False, "error": "not found"}, 404)
        if not self._authed():
            return self._send_json({"ok": False, "error": "unauthorized"}, 401)
        req = self._read_json()
        if req is None:
            return self._send_json({"ok": False, "error": "bad request"}, 400)
        if path == "/call":
            name = req.get("name")
            if not name:
                return self._send_json({"ok": False, "error": "missing tool name"}, 400)
            return self._send_json(self.server.tools.call(name, req.get("arguments", {})))
        if path == "/ops/run":                         # launch a maintenance verb
            try:
                return self._send_json(
                    self.server.ops.start(req.get("command", ""), req.get("args") or {}))
            except ValueError as e:                    # unknown verb / bad option
                return self._send_json({"ok": False, "error": str(e)}, 400)
        if path == "/ops/stop":
            return self._send_json(self.server.ops.stop())
        if path == "/ops/reload":                      # re-warm caches after a write-job / crash
            kb = getattr(self.server, "kb", None)
            if kb is None:
                return self._send_json({"ok": False, "error": "no KB loaded"}, 400)
            try:
                return self._send_json({"ok": True, "counts": kb.reload()})
            except Exception as e:                     # pragma: no cover - defensive
                return self._send_json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)
        if path == "/ops/autopilot":                   # Prioritizer tab: save the plan
            from . import autopilot as A
            try:
                saved = A.save_plan(self.cfg, req.get("plan") or {})
            except (ValueError, OSError) as e:
                return self._send_json({"ok": False, "error": str(e)}, 400)
            ap = getattr(self.server, "autopilot", None)
            if ap is not None:                         # apply enable/disable live
                ap.start() if saved["enabled"] else None
            return self._send_json({"ok": True, "plan": saved})
        if path == "/config":                          # persist scalar settings to config.toml
            from .config import update_config_file
            cp = self.cfg.get("_config_path")
            if not cp:
                return self._send_json(
                    {"ok": False, "error": "server started without -c; no config file to write"}, 400)
            try:
                applied = update_config_file(cp, req.get("updates") or {})
            except (ValueError, OSError) as e:
                return self._send_json({"ok": False, "error": str(e)}, 400)
            return self._send_json({"ok": True, "applied": applied, "note":
                "saved — restart to apply, or 'Reload KB' for read-path keys"})
        if path == "/library/config":                  # persist WHICH subfolders are indexed
            from .config import (resolve_library_selection, write_library_sources,
                                 library_status)
            cp = self.cfg.get("_config_path")
            if not cp:
                return self._send_json(
                    {"ok": False, "error": "server started without -c; no config file to write"}, 400)
            try:
                paths = resolve_library_selection(self.cfg, req.get("active") or [])
                write_library_sources(cp, paths)       # containment-validated names only
            except (ValueError, OSError) as e:
                return self._send_json({"ok": False, "error": str(e)}, 400)
            self.cfg["library_sources"] = paths        # reflect immediately for the next GET
            return self._send_json({"ok": True, **library_status(self.cfg), "note":
                "saved — run 'ingest-library' in Operations to (re)index the selection"})
        if path == "/source":                          # rename / regroup / relicense (on the MASTER)
            doc_id = req.get("doc_id")
            if not doc_id:
                return self._send_json({"ok": False, "error": "missing doc_id"}, 400)
            # only pass fields the client actually sent (None = don't touch)
            kw = {k: req[k] for k in ("title", "bundle", "license", "license_holder",
                                      "license_url", "license_text") if k in req}
            mkb = self.server.open_master_kb()
            try:
                row = mkb.set_source(doc_id, **kw)
                if row is not None:
                    row["shippable_flags"] = mkb.license_of(doc_id)["flags"]
            finally:
                mkb.close()
            if row is None:
                return self._send_json({"ok": False, "error": f"no such source: {doc_id}"}, 404)
            return self._send_json({"ok": True, "source": row, "note":
                "saved to master — Apply a scenario (or restart) to fold into the live session"})
        if path == "/scenario":                        # switch scenario + hot-swap the KB
            try:
                info = self.server.swap_scenario(req.get("scenario"))
            except Exception as e:
                return self._send_json(
                    {"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)
            return self._send_json({"ok": True, **info})


class KnowledgeHostServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, cfg, store, tools, kb=None):
        self.cfg = cfg
        self.store = store
        self.tools = tools
        self.kb = kb
        self.ops = OpsRunner(cfg)                   # single-slot maintenance-job runner
        from . import autopilot as _A
        from . import lm_lease as _L
        self.autopilot = _A.Autopilot(cfg, self.ops, lease_mod=_L)   # priority-driven verb runner
        self._swap_lock = __import__("threading").Lock()
        super().__init__((cfg["host"], cfg["port"]), Handler)

    def master_kb_path(self) -> str:
        return self.cfg.get("_master_kb_path") or self.cfg["kb_path"]

    def open_master_kb(self):
        """A short-lived KB handle on the MASTER (not the served working copy) for admin
        edits like renaming/regrouping a source — changes must land in the authoring source
        of truth, not the disposable session DB.  Caller must close it."""
        from .kb import KB
        return KB({**self.cfg, "kb_path": self.master_kb_path(), "ann_search": False})

    def swap_scenario(self, name: str | None = None) -> dict:
        """Switch the live session to a scenario: reassemble its working DB from the master
        and hot-swap the KB the server + tools read from (old handle closed).  Serialised so
        two panel clicks can't race a half-built swap."""
        from . import bundles
        from .kb import KB
        with self._swap_lock:
            if name:
                self.cfg["active_scenario"] = name
            self.cfg["kb_path"] = self.master_kb_path()      # assemble from the master
            work = bundles.assemble_working_db(self.cfg, force=True)
            if work != self.cfg["kb_path"]:
                self.cfg["kb_path"] = work
                self.cfg["ann_path"] = ""
            new_kb = KB(self.cfg)
            try:
                new_kb.warm()
                new_kb._get_ann()
            except Exception:                                # pragma: no cover - best effort
                pass
            old, self.kb, self.tools.kb = self.kb, new_kb, new_kb
            if old is not None:
                try:
                    old.close()
                except Exception:                            # pragma: no cover
                    pass
            return {"scenario": bundles.active_scenario_name(self.cfg),
                    "working_db": work, "counts": new_kb.counts()}


def serve(cfg, store, tools, kb=None):
    httpd = KnowledgeHostServer(cfg, store, tools, kb)
    # Warm the ANN index now (one-time resident load of ~index-size RAM) so the first
    # `ask` doesn't eat the load — every query is RAM-speed from the first one.
    if kb is not None:
        try:
            ann = kb._get_ann()
            if ann is not None:
                log.info("ANN warmed: %d node vectors resident", len(ann))
        except Exception as e:
            log.warning("ANN warm failed (%s) — falling back to brute force", e)
        # Pull the node/card tables into the page cache so the first ask's candidate
        # hydration doesn't fault hundreds of rows from disk.
        import time as _t
        _t0 = _t.perf_counter()
        kb.warm()
        log.info("KB tables warmed into page cache (%.1fs)", _t.perf_counter() - _t0)
    log.info("listening on http://%s:%s (backend=%s, %d chunks, dense=%s)",
             cfg["host"], cfg["port"], store.backend, store.count(),
             store.has_vectors())
    if cfg.get("auth_token"):
        log.info("auth: Bearer token required on /call")
    # Autopilot: start the thread; it no-ops until the saved plan is enabled (Prioritizer tab).
    try:
        httpd.autopilot.start()
        from . import autopilot as _A
        if _A.load_plan(cfg).get("enabled"):
            log.info("autopilot: enabled — running maintenance verbs on a priority basis")
    except Exception as e:                              # pragma: no cover
        log.warning("autopilot failed to start (%s) — maintenance stays manual", e)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        try:
            httpd.autopilot.stop()                 # stop the priority driver before its job runner
        except Exception:                          # pragma: no cover
            pass
        httpd.ops.shutdown()                       # don't leave a job orphaned past the server
        httpd.shutdown()
        store.close()
        lib = getattr(tools, "library_store", None)
        if lib is not None:
            try:
                lib.close()
            except Exception:                      # pragma: no cover
                pass
        if kb is not None:
            kb.close()
