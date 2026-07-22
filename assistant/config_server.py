#!/usr/bin/env python
"""
Config web service — a localhost-only editor + inspector for the Vinkona cascade.

Serves a single-page UI plus a small REST API.  Stdlib only (no deps).  Binds to
config_server.host (127.0.0.1 by default) since it edits prompts/settings and the
memory store with no auth.

Endpoints
  GET  /                     the UI
  GET/POST /api/config       config/config.json (whole document)
  GET/POST /api/personas     config/personas.json (whole document)
  GET  /api/models?tier=…    list GGUFs in models_dir + this tier's llama.cpp settings
  GET  /api/trace?n=…        recent fast/big-LM activity (the live feed)
  GET  /api/memory           list memory entries (SQLite)
  POST /api/memory           upsert one entry (recomputes its embedding)
  POST /api/memory/delete    delete one entry by id
  GET  /api/self             Vinkona's self-authored identity (traits) + self-memories
  POST /api/self/attribute   set/edit one identity attribute
  POST /api/self/attribute/delete   delete one identity attribute by id
  POST /api/self/revert      drop the adaptations grown over one core trait

Edits to config/personas land in the JSON files; the cascade re-reads personas +
tunables per connection, so those apply on the next call.  Structural changes
(ports, models, the ASR model) need a service restart.  Memory edits hit the DB
directly and are picked up on the cascade's next connection (it reloads per call).

  python config_server.py --config config/config.json
"""

import argparse
import array
import importlib.util
import json
import math
import os
import sqlite3
import time
import urllib.request
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _load_mod(name: str):
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).parent / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _atomic_json(path: Path, obj) -> None:
    """Write JSON via temp + os.replace.  config.json/personas.json are read concurrently
    by the supervisor, cascade, worker and llm_server — a truncate-then-write could hand
    any of them half a file (which load_config 'survives' by silently running on pure
    DEFAULTS), and a crash mid-write would corrupt the file durably."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


CFGMOD = _load_mod("config")
PEOPLE = _load_mod("people")           # for the Self tab (Vinkona's self-authored identity)
IDLECTL = _load_mod("idle_control")    # idle pause/resume + quiet-hours math
HELPMOD = _load_mod("confighelp")      # /api/help — extracted config.py comments + help.json
UI_PATH = Path(__file__).parent / "config_ui.html"
LOGS_DIR = Path(__file__).parent / "logs"            # written by vinkona.sh (shared filesystem)


def _research_defaults() -> dict:
    """The built-in prompt defaults from memory.py, so the Research tab can show
    '(default)' placeholder text and let the user reset a field to it.  Loaded lazily
    (memory.py imports numpy) and best-effort — returns {} if it can't be read."""
    try:
        spec = importlib.util.spec_from_file_location(
            "memory", str(Path(__file__).parent / "memory.py"))
        mem = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mem)
        return {"research_prompt": mem.DEFAULT_RESEARCH_PROMPT,
                "synth_prompt": mem.DEFAULT_SYNTH_PROMPT,
                "ingest_prompt": mem.DEFAULT_INGEST_PROMPT,
                "reflection_prompt": mem.DEFAULT_REFLECTION_PROMPT,
                "introspect_prompt": mem.DEFAULT_IDLE_REFLECT_PROMPT,
                "consolidate_prompt": mem.DEFAULT_CONSOLIDATE_PROMPT}
    except Exception as e:
        return {"error": str(e)}


def _briefing_default() -> str:
    """The built-in big-LM briefing prompt (llm_bridge.DEFAULT_BRIEFING_PROMPT), so the LM
    tab can show it as placeholder and offer a copy-paste-able reset.  Best-effort — returns
    '' if it can't be read (llm_bridge imports aiohttp)."""
    try:
        return _load_mod("llm_bridge").DEFAULT_BRIEFING_PROMPT
    except Exception:
        return ""


def _tail(path: Path, n: int) -> str:
    """Last n lines of a (possibly large) file, without reading it all."""
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block, data, lines = 8192, b"", 0
            while size > 0 and lines <= n:
                step = min(block, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
                lines = data.count(b"\n")
            return b"\n".join(data.splitlines()[-n:]).decode("utf8", "replace")
    except Exception:
        return ""

# UI tier name → config key.
TIER_KEY = {"fast": "fast_lm", "big": "big_lm", "embed": "embed_lm"}


# ── Embedding helper (OpenAI /v1/embeddings, e.g. llama.cpp --embedding) ──────

def _embed_blob(base_url: str, model: str, text: str) -> bytes | None:
    """Return a normalized float32 embedding blob (matches memory.py), or None."""
    payload = json.dumps({"model": model, "input": text}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/embeddings", data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read()).get("data") or [{}]
            vec = data[0].get("embedding", [])
    except Exception:
        return None
    if not vec:
        return None
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0:
        return None
    return array.array("f", [x / norm for x in vec]).tobytes()


# ── Memory store access (raw SQLite; no numpy needed for browse/edit) ────────

class MemoryAdmin:
    FIELDS = ("id", "triggers", "context_tags", "payload", "priority", "recency",
              "last_used", "created_at", "category", "expiry", "source", "cooldown_until",
              "doc_id", "status")

    def __init__(self, cfg: dict):
        self.m = cfg["memory"]
        self.e = cfg["embed_lm"]
        self.path = self.m["db_path"]
        if self.e.get("remote"):     # remote embed tier: agree on the model NAME
            try:
                CFGMOD.resolve_remote_lms(cfg)
            except Exception:
                pass

    # Matches memory.py's schema, so the UI can seed entries before the cascade
    # (which owns the store) has ever created the DB.
    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY, triggers TEXT, context_tags TEXT, payload TEXT,
            priority INTEGER, recency REAL, last_used REAL, created_at REAL,
            category TEXT, expiry REAL, source TEXT, cooldown_until REAL, embedding BLOB,
            doc_id TEXT);
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, ts REAL, role TEXT, text TEXT);
        CREATE INDEX IF NOT EXISTS idx_logs_session ON chat_logs(session_id);"""

    def _conn(self, ensure=False):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(self.path, timeout=5)
        c.row_factory = sqlite3.Row
        if ensure:
            c.executescript(self._SCHEMA)
        return c

    def list(self) -> list[dict]:
        if not Path(self.path).exists():
            return []
        with self._conn() as c:
            rows = c.execute("SELECT * FROM memories ORDER BY priority DESC, created_at DESC")
            out = []
            for row in rows:
                cols = row.keys()
                e = {k: (row[k] if k in cols else None) for k in self.FIELDS}
                e["triggers"] = json.loads(e["triggers"] or "[]")
                e["context_tags"] = json.loads(e["context_tags"] or "[]")
                e["has_embedding"] = row["embedding"] is not None
                out.append(e)
            return out

    def document(self, doc_id: str) -> dict | None:
        if not Path(self.path).exists():
            return None
        with self._conn() as c:
            try:
                row = c.execute("SELECT id,url,title,topic,fetched_at,text FROM documents "
                                "WHERE id=?", (doc_id,)).fetchone()
            except sqlite3.OperationalError:
                return None                       # no documents table yet
            return dict(row) if row else None

    def plans(self, status: str = "all", limit: int = 200) -> dict:
        """Learning plans with their questions, for the Plans view.  `status` filters
        (all|open|done); counts are over ALL plans (so completed ones aren't hidden by the
        display limit).  'done' shows most-recently-completed first; else newest-created first."""
        empty = {"plans": [], "counts": {"total": 0, "open": 0, "done": 0}, "shown": 0}
        if not Path(self.path).exists():
            return empty
        with self._conn() as c:
            try:
                counts = {"total": 0, "open": 0, "done": 0}
                for row in c.execute("SELECT status, COUNT(*) n FROM learning_plans GROUP BY status"):
                    counts["done" if row["status"] == "done" else "open"] += row["n"]
                    counts["total"] += row["n"]
                where, params = "", []
                if status == "open":
                    where = "WHERE status!='done'"
                elif status == "done":
                    where = "WHERE status='done'"
                order = "completed_at DESC, id DESC" if status == "done" else "id DESC"
                plans = [dict(r) for r in c.execute(
                    f"SELECT * FROM learning_plans {where} ORDER BY {order} LIMIT ?",
                    (*params, int(limit)))]
                for p in plans:
                    p["questions"] = [dict(r) for r in c.execute(
                        "SELECT id,question,kind,status,answer,updated_at FROM plan_questions "
                        "WHERE plan_id=? ORDER BY id", (p["id"],))]
            except sqlite3.OperationalError:
                return empty                          # no plans tables yet
            return {"plans": plans, "counts": counts, "shown": len(plans)}

    # A dead-end research question is one answered with a stock failure: '(no source found)' from
    # the fetcher, or the big LM's own 'the source does not answer this' verdict.  Re-opening puts
    # it back in the pile so the worker retries it (e.g. after the web-search fix landed).
    _DEADENDS = {
        "no_source": ("answer = ?", ("(no source found)",)),
        "no_answer": ("LOWER(answer) LIKE ?", ("%does not answer%",)),
    }

    def research_deadend_counts(self) -> dict:
        """How many research questions sit in each dead-end bucket (for the reset buttons)."""
        out = {"no_source": 0, "no_answer": 0}
        if not Path(self.path).exists():
            return out
        with self._conn() as c:
            for which, (pred, params) in self._DEADENDS.items():
                try:
                    out[which] = c.execute(
                        f"SELECT COUNT(*) FROM plan_questions WHERE kind='research' "
                        f"AND status='answered' AND {pred}", params).fetchone()[0]
                except sqlite3.OperationalError:
                    return out                        # no plan tables yet
        return out

    def reopen_research_questions(self, which: str) -> dict:
        """Reset one dead-end bucket back to 'open' (answer cleared) and re-open any parent plan
        that had been marked done, so the worker researches them again.  Returns counts."""
        import time
        spec = self._DEADENDS.get(which)
        if spec is None:
            return {"ok": False, "error": f"unknown reset '{which}'"}
        if not Path(self.path).exists():
            return {"ok": False, "error": "no memory db yet"}
        pred, params = spec
        with self._conn(ensure=True) as c:
            try:
                rows = c.execute(
                    f"SELECT id, plan_id FROM plan_questions WHERE kind='research' "
                    f"AND status='answered' AND {pred}", params).fetchall()
                ids = [r["id"] for r in rows]
                plan_ids = sorted({r["plan_id"] for r in rows})
                plans_reopened = 0
                if ids:
                    c.execute(
                        f"UPDATE plan_questions SET status='open', answer=NULL, updated_at=? "
                        f"WHERE id IN ({','.join('?' * len(ids))})", (time.time(), *ids))
                    cur = c.execute(                  # rowcount = plans that actually flip done→open
                        f"UPDATE learning_plans SET status='open', completed_at=NULL "
                        f"WHERE status='done' AND id IN ({','.join('?' * len(plan_ids))})",
                        tuple(plan_ids))
                    plans_reopened = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
                c.commit()
            except sqlite3.OperationalError as e:
                return {"ok": False, "error": str(e)}
        return {"ok": True, "reopened": len(ids), "plans_reopened": plans_reopened}

    def reopen_plan(self, plan_id, which: str = "all") -> dict:
        """Reset ONE learning plan: put its questions back to 'open' (answers cleared) and re-open
        the plan, so idle researches it again.  which='research' resets only research questions
        (leaves the ask-the-user ones); 'all' resets every question.  Returns the count reopened."""
        import time
        if not Path(self.path).exists():
            return {"ok": False, "error": "no memory db yet"}
        try:
            pid = int(plan_id)
        except (TypeError, ValueError):
            return {"ok": False, "error": "bad plan id"}
        kind_pred = "" if which == "all" else "AND kind='research'"
        with self._conn(ensure=True) as c:
            try:
                cur = c.execute(
                    f"UPDATE plan_questions SET status='open', answer=NULL, updated_at=? "
                    f"WHERE plan_id=? AND status!='open' {kind_pred}", (time.time(), pid))
                reopened = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
                c.execute("UPDATE learning_plans SET status='open', completed_at=NULL WHERE id=?",
                          (pid,))
                c.commit()
            except sqlite3.OperationalError as e:
                return {"ok": False, "error": str(e)}
        return {"ok": True, "reopened": reopened}

    def upsert(self, e: dict) -> dict:
        import time, uuid
        now = time.time()
        mid = (e.get("id") or "").strip() or uuid.uuid4().hex[:12]
        triggers = [t.strip() for t in e.get("triggers", []) if t.strip()]
        tags = [t.strip() for t in e.get("context_tags", []) if t.strip()]
        payload = e.get("payload", "")
        blob = _embed_blob(self.e["url"], self.e["model"],
                           " ".join(triggers) + " " + payload)
        with self._conn(ensure=True) as c:
            existing = c.execute("SELECT 1 FROM memories WHERE id=?", (mid,)).fetchone()
            if existing:
                c.execute("""UPDATE memories SET triggers=?, context_tags=?, payload=?,
                             priority=?, category=?, expiry=?, source=?, embedding=?
                             WHERE id=?""",
                          (json.dumps(triggers), json.dumps(tags), payload,
                           int(e.get("priority", 1)), e.get("category", "general"),
                           e.get("expiry"), e.get("source", "manual"), blob, mid))
            else:
                c.execute("""INSERT INTO memories
                             (id,triggers,context_tags,payload,priority,recency,last_used,
                              created_at,category,expiry,source,cooldown_until,embedding)
                             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                          (mid, json.dumps(triggers), json.dumps(tags), payload,
                           int(e.get("priority", 1)), 0.0, 0.0, now,
                           e.get("category", "general"), e.get("expiry"),
                           e.get("source", "manual"), 0.0, blob))
            c.commit()
        return {"ok": True, "id": mid, "embedded": blob is not None}

    def delete(self, mid: str):
        if not Path(self.path).exists():
            return
        with self._conn(ensure=True) as c:
            c.execute("DELETE FROM memories WHERE id=?", (mid,))
            c.commit()

    def quarantined(self) -> "list[dict]":
        """Quarantined memories (reconcile fragments), so the UI can review/restore them."""
        if not Path(self.path).exists():
            return []
        with self._conn() as c:
            try:
                rows = c.execute("SELECT * FROM memories WHERE status='quarantined' "
                                 "ORDER BY created_at DESC").fetchall()
            except sqlite3.OperationalError:
                return []                             # no status column yet
            out = []
            for row in rows:
                e = {k: (row[k] if k in row.keys() else None) for k in self.FIELDS}
                e["triggers"] = json.loads(e["triggers"] or "[]")
                e["context_tags"] = json.loads(e["context_tags"] or "[]")
                out.append(e)
            return out

    def set_status(self, mid: str, status: str) -> dict:
        """Restore ('active' → NULL) or quarantine a memory by id."""
        if not Path(self.path).exists():
            return {"ok": False}
        val = None if status == "active" else "quarantined"
        with self._conn(ensure=True) as c:
            try:
                c.execute("UPDATE memories SET status=? WHERE id=?", (val, mid))
            except sqlite3.OperationalError:
                c.execute("ALTER TABLE memories ADD COLUMN status TEXT")
                c.execute("UPDATE memories SET status=? WHERE id=?", (val, mid))
            c.commit()
        return {"ok": True, "id": mid, "status": status}

    def request_reconcile(self) -> dict:
        """Queue a personal-fact reconcile for the research worker (it polls worker_state)."""
        if not Path(self.path).exists():
            return {"ok": False, "error": "no memory db yet"}
        with self._conn(ensure=True) as c:
            c.execute("CREATE TABLE IF NOT EXISTS worker_state (key TEXT PRIMARY KEY, value TEXT)")
            c.execute("INSERT INTO worker_state(key,value) VALUES('reconcile_request',?) "
                      "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(time.time()),))
            c.commit()
        return {"ok": True, "queued": True}

    def idle_status(self, cfg: dict) -> dict:
        """Effective idle-work state: the manual override (worker_state) resolved against
        the scheduled quiet hours (config), for the header button + Settings."""
        _ic = IDLECTL
        override = ""
        try:
            if Path(self.path).exists():
                with self._conn() as c:
                    row = c.execute("SELECT value FROM worker_state WHERE key='idle_override'").fetchone()
                    if row:
                        override = row[0] or ""
        except Exception:
            pass
        quiet = (cfg.get("research", {}).get("idle", {}) or {}).get("quiet_hours", []) or []
        now_min = _ic.now_minutes(time.localtime())
        d = _ic.describe(override, now_min, quiet)
        d["quiet_hours"] = quiet
        return d

    def set_idle_override(self, override: str) -> dict:
        """Set the manual pause/resume switch the worker polls: 'paused' | 'active' | 'auto'."""
        override = (override or "auto").strip().lower()
        if override not in ("paused", "active", "auto"):
            return {"ok": False, "error": "override must be paused | active | auto"}
        stored = "" if override == "auto" else override
        with self._conn(ensure=True) as c:
            c.execute("CREATE TABLE IF NOT EXISTS worker_state (key TEXT PRIMARY KEY, value TEXT)")
            c.execute("INSERT INTO worker_state(key,value) VALUES('idle_override',?) "
                      "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (stored,))
            c.commit()
        return {"ok": True, "override": override}

    def request_export(self) -> dict:
        """Queue a FULL research export (documents -> solved/*.md) for the worker — it polls
        worker_state and rebuilds every drop, repairing anything removed from the folder."""
        if not Path(self.path).exists():
            return {"ok": False, "error": "no memory db yet"}
        with self._conn(ensure=True) as c:
            c.execute("CREATE TABLE IF NOT EXISTS worker_state (key TEXT PRIMARY KEY, value TEXT)")
            c.execute("INSERT INTO worker_state(key,value) VALUES('export_request',?) "
                      "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(time.time()),))
            c.commit()
        return {"ok": True, "queued": True}

    def reembed(self) -> dict:
        """Recompute the embedding for every entry (e.g. after starting serve_embed)."""
        if not Path(self.path).exists():
            return {"embedded": 0, "total": 0}
        ok = total = 0
        with self._conn(ensure=True) as c:
            rows = c.execute("SELECT id, triggers, payload FROM memories").fetchall()
            for r in rows:
                total += 1
                triggers = json.loads(r["triggers"] or "[]")
                blob = _embed_blob(self.e["url"], self.e["model"],
                                   " ".join(triggers) + " " + (r["payload"] or ""))
                if blob is not None:
                    c.execute("UPDATE memories SET embedding=? WHERE id=?", (blob, r["id"]))
                    ok += 1
            c.commit()
        return {"embedded": ok, "total": total}


# ── Self / personality access (people store + self-memories) ─────────────────
class SelfAdmin:
    """Read/edit Vinkona's self-authored personality: the structured identity store
    (HEXACO-style traits in core/compensated/surface layers, with provenance) and her
    category='self' memories.  Owner-level: the config UI is the user, so it may edit even
    core/locked canon (the conversation-only restriction exists to keep UNTRUSTED data
    out, not the owner)."""

    def __init__(self, cfg: dict):
        self.path = cfg["memory"]["db_path"]

    def _conn(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(self.path, timeout=5)
        c.row_factory = sqlite3.Row
        return c

    def view(self) -> dict:
        if not Path(self.path).exists():
            return {"self": None, "attributes": [], "self_memories": []}
        c = self._conn()
        try:
            ps = PEOPLE.PeopleStore(c)
            s = ps.by_kind("self")
            attrs = ps.attributes(s["id"]) if s else []
            mems = []
            try:
                for r in c.execute(
                        "SELECT id,payload,priority,triggers,context_tags,source,created_at "
                        "FROM memories WHERE category='self' "
                        "ORDER BY priority DESC, created_at DESC"):
                    d = dict(r)
                    d["triggers"] = json.loads(d["triggers"] or "[]")
                    d["context_tags"] = json.loads(d["context_tags"] or "[]")
                    mems.append(d)
            except sqlite3.OperationalError:
                pass
            return {"self": dict(s) if s else None, "attributes": attrs, "self_memories": mems,
                    "state": ps.self_state(), "state_history": ps.self_state_history(12),
                    # adaptations carry the core they are cast from, so the panel
                    # can show the grounding rather than a detached second self
                    "adaptations": ps.adaptations(s["id"]) if s else [],
                    "presented": [ps._cast(a) for a in ps.effective(s["id"])
                                  if a["facet"] == "trait"] if s else [],
                    # her reflection record — applied/refused/deferred/no-change, with
                    # the reasoning.  Collapsed in the panel: it's for perusal and
                    # debugging, not something to put in the owner's face.
                    "reflections": ps.trait_decisions(25) if s else [],
                    # the persona's own pronouns, so the panel's prose follows the
                    # character the user chose rather than assuming one
                    "pronouns": ps.pronoun_set()}
        finally:
            c.close()

    def set_state(self, obj: dict) -> dict:
        c = self._conn()
        try:
            PEOPLE.PeopleStore(c).set_self_state(obj.get("text", ""), source="user_edit")
            return {"ok": True}
        finally:
            c.close()

    def set_attribute(self, obj: dict) -> dict:
        c = self._conn()
        try:
            ps = PEOPLE.PeopleStore(c)
            s = ps.by_kind("self")
            pid = s["id"] if s else ps.ensure_person("self", name="Vinkona")
            layer = obj.get("layer") or "core"
            if layer == "compensated":
                # even owner-level edits go through the guards: an adaptation with
                # no grounding or no context isn't an adaptation, whoever wrote it
                try:
                    ps.adapt(pid, (obj.get("key") or "").strip(),
                             (obj.get("value") or "").strip(),
                             context=obj.get("context") or "",
                             derived_from=obj.get("derived_from") or "",
                             mode=obj.get("mode") or "expresses",
                             facet=(obj.get("facet") or "trait").strip(),
                             provenance="user_edit")
                except ValueError as e:
                    return {"ok": False, "error": str(e)}
                return {"ok": True}
            ps.set_attribute(pid, (obj.get("facet") or "trait").strip(),
                             (obj.get("key") or "").strip(), (obj.get("value") or "").strip(),
                             layer=layer, provenance="user_edit",
                             locked=obj.get("locked"))
            return {"ok": True}
        finally:
            c.close()

    def revert_to_core(self, obj: dict) -> dict:
        """Drop the adaptations grown over one core trait — she falls back to canon."""
        c = self._conn()
        try:
            ps = PEOPLE.PeopleStore(c)
            s = ps.by_kind("self")
            if not s:
                return {"ok": False, "error": "no self record"}
            n = ps.revert_to_core(s["id"], (obj.get("key") or "").strip())
            return {"ok": True, "reverted": n}
        finally:
            c.close()

    def delete_attribute(self, obj: dict) -> dict:
        c = self._conn()
        try:
            PEOPLE.PeopleStore(c).delete_attribute(int(obj.get("id")))
            return {"ok": True}
        finally:
            c.close()


# ── HTTP handler ─────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    config_path = "config/config.json"

    def _send(self, code, body, ctype="application/json"):
        if not isinstance(body, (bytes, bytearray)):
            body = str(body).encode("utf8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj))

    def _cfg(self):
        return CFGMOD.load_config(self.config_path)

    def _personas_path(self):
        return self._cfg().get("personas_path", "config/personas.json")

    def _query(self):
        q = urllib.parse.urlparse(self.path).query
        return urllib.parse.parse_qs(q)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def _local_only(self) -> bool:
        """Defeat DNS-rebinding / CSRF.  This service is localhost-only and unauthenticated,
        and it serves an HTML UI plus endpoints that expose config + the trace feed (which
        carries assembled prompts and recalled personal memory) and can edit prompts / fire
        restarts.  A browser on a malicious page must not be able to reach it, so: require the
        Host header to name loopback (a rebinding attack carries the attacker's hostname), and
        reject any cross-origin request.  Answers 403 and returns False when it looks remote."""
        ok = {"127.0.0.1", "localhost", "::1", "[::1]",
              str(self.server.server_address[0])}          # also the actual bind address
        host = (self.headers.get("Host") or "").strip().lower()
        hostname = host.rsplit(":", 1)[0] if host else ""
        if hostname not in ok:
            self._json(403, {"error": "forbidden (non-local Host)"})
            return False
        origin = self.headers.get("Origin")
        if origin:
            oh = (urllib.parse.urlparse(origin).hostname or "").lower()
            if oh not in ok:
                self._json(403, {"error": "forbidden (cross-origin)"})
                return False
        return True

    # ── GET ──────────────────────────────────────────────────────────────────
    def do_GET(self):
        if not self._local_only():
            return
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, UI_PATH.read_text(), "text/html; charset=utf-8")
        if path == "/api/config":
            # The MERGED config (defaults ⊕ user overrides), so the UI always shows every
            # current option even if config.json predates a schema addition.  Runtime
            # resolution (profile paths) is intentionally NOT applied here.
            return self._send(200, json.dumps(CFGMOD.merged_config(self.config_path), indent=2))
        if path == "/api/help":
            # Help text keyed by dotted config path (extracted from config.py's own
            # comments) plus "tab.*" intros from help.json — see confighelp.py.
            # Both re-read on change, so editing help needs only a browser refresh.
            try:
                return self._json(200, {"help": HELPMOD.load(Path(__file__).parent)})
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/net":                        # the egress broker's window
            try:
                netadmin = _load_mod("netadmin")
                return self._json(200, netadmin.view(self._cfg()))
            except Exception as e:
                return self._json(200, {"ok": False, "error": f"{type(e).__name__}: {e}"})
        if path == "/api/personas":
            return self._send(200, CFGMOD.resolve_read(self._personas_path()).read_text())
        if path == "/api/models":
            return self._get_models()
        if path == "/api/trace":
            return self._get_trace()
        if path == "/api/services":
            logs = sorted(LOGS_DIR.glob("*.log")) if LOGS_DIR.is_dir() else []
            return self._json(200, {"services": [
                {"name": p.stem, "size": p.stat().st_size, "mtime": p.stat().st_mtime}
                for p in logs if not p.stem.startswith("_")]})
        if path == "/api/vinur":
            # The knowledge-host connection badge: /health + the authed /drop
            # handshake (token validity, drops held, open gaps).
            try:
                link = _load_mod("supervisor").vinur_link(
                    CFGMOD.load_config(self.config_path))
            except Exception as e:
                return self._json(200, {"configured": False, "error": str(e)})
            if not link:
                return self._json(200, {"configured": False})
            return self._json(200, {"configured": True, **link})
        if path == "/api/logs":
            q = self._query()
            name = (q.get("service", [""])[0]).replace("/", "").replace("..", "")
            n = int(q.get("n", ["200"])[0])
            log = LOGS_DIR / f"{name}.log"
            if not name or not log.exists():
                return self._json(404, {"error": "no such service log"})
            return self._json(200, {"service": name, "text": _tail(log, n)})
        if path == "/api/memory":
            try:
                return self._json(200, {"entries": MemoryAdmin(self._cfg()).list()})
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/document":
            doc_id = self._query().get("id", [""])[0]
            try:
                doc = MemoryAdmin(self._cfg()).document(doc_id)
                return self._json(200 if doc else 404, doc or {"error": "not found"})
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/plans":
            try:
                q = self._query()
                status = (q.get("status", ["all"])[0] or "all").lower()
                limit = int((q.get("limit", ["200"])[0] or "200"))
                return self._json(200, MemoryAdmin(self._cfg()).plans(status, limit))
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/quarantined":
            try:
                return self._json(200, {"entries": MemoryAdmin(self._cfg()).quarantined()})
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/self":
            try:
                return self._json(200, SelfAdmin(self._cfg()).view())
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/idle":
            try:
                return self._json(200, MemoryAdmin(self._cfg()).idle_status(self._cfg()))
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/research_defaults":
            return self._json(200, _research_defaults())
        if path == "/api/research/deadends":
            try:
                return self._json(200, MemoryAdmin(self._cfg()).research_deadend_counts())
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/mode":
            mf = LOGS_DIR / "control" / "mode"
            mode = "normal"
            try:
                if mf.exists():
                    mode = mf.read_text().strip() or "normal"
            except Exception:
                pass
            if mode not in ("normal", "knowledge"):
                mode = "normal"
            return self._json(200, {"mode": mode, "modes": ["normal", "knowledge"],
                                    "big_lm2_ctx": (self._cfg().get("big_lm2") or {}).get("ctx_size", 4096)})
        if path == "/api/profiles":
            return self._json(200, {"active": CFGMOD.active_profile(),
                                    "profiles": [CFGMOD.profile_stats(n)
                                                 for n in CFGMOD.list_profiles()]})
        self._json(404, {"error": "not found"})

    def _get_models(self):
        cfg = self._cfg()
        tier = self._query().get("tier", ["fast"])[0]
        key = TIER_KEY.get(tier, "fast_lm")
        block = cfg.get(key, {})
        models_dir = cfg.get("models_dir", "Models")
        ggufs = sorted(p.name for p in Path(models_dir).glob("*.gguf")) if Path(models_dir).is_dir() else []
        settings = {
            "url": block.get("url"), "model": block.get("model"),
            "gpu": block.get("gpu", 0), "ctx_size": block.get("ctx_size", 4096),
            "n_gpu_layers": block.get("n_gpu_layers", 99),
            "flash_attn": bool(block.get("flash_attn", False)),
            "extra_args": block.get("extra_args", []),
            "lead": block.get("lead", 1),              # big tier: how much it drives the chat
        }
        if key == "big_lm":                            # editable briefing prompt + its default
            settings["briefing_prompt"] = block.get("briefing_prompt") or ""
            settings["briefing_default"] = _briefing_default()
        self._json(200, {"tier": tier, "key": key, "models_dir": models_dir,
                         "models": ggufs, "settings": settings})

    def _get_trace(self):
        cfg = self._cfg()
        cs = cfg["config_server"]
        n = int(self._query().get("n", ["200"])[0])
        path = Path(cs.get("trace_path", "config/trace.jsonl"))
        events = []
        if path.exists():
            for line in path.read_text().splitlines()[-n:]:
                try:
                    events.append(json.loads(line))
                except Exception:
                    continue
        self._json(200, {"events": events})

    # ── POST ─────────────────────────────────────────────────────────────────
    def do_POST(self):
        if not self._local_only():
            return
        path = urllib.parse.urlparse(self.path).path
        try:
            obj = self._body()                       # validate JSON before acting
        except Exception as e:
            return self._json(400, {"error": f"invalid JSON: {e}"})
        if path == "/api/config":
            _atomic_json(Path(self.config_path), obj)
            return self._json(200, {"ok": True})
        if path == "/api/net":                        # broker action (revoke / rule toggle)
            try:
                return self._json(200, _load_mod("netadmin").action(obj))
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/net/setting":                # save one redacted network setting
            try:
                netadmin = _load_mod("netadmin")
                return self._json(200, netadmin.set_setting(
                    self.config_path, str(obj.get("key") or ""), obj.get("value")))
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/personas":
            _atomic_json(Path(self._personas_path()), obj)
            return self._json(200, {"ok": True})
        if path == "/api/memory":
            try:
                return self._json(200, MemoryAdmin(self._cfg()).upsert(obj))
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/memory/status":
            try:
                return self._json(200, MemoryAdmin(self._cfg()).set_status(
                    obj.get("id", ""), obj.get("status", "active")))
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/reconcile":
            try:
                return self._json(200, MemoryAdmin(self._cfg()).request_reconcile())
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/research/export":
            try:
                return self._json(200, MemoryAdmin(self._cfg()).request_export())
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/idle":
            try:
                return self._json(200, MemoryAdmin(self._cfg()).set_idle_override(
                    obj.get("override", "auto")))
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/research/reopen":
            try:
                return self._json(200, MemoryAdmin(self._cfg())
                                  .reopen_research_questions(obj.get("which", "")))
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/plan/reopen":
            try:
                return self._json(200, MemoryAdmin(self._cfg())
                                  .reopen_plan(obj.get("plan_id"), obj.get("which", "all")))
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/memory/delete":
            try:
                MemoryAdmin(self._cfg()).delete(obj.get("id", ""))
                return self._json(200, {"ok": True})
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/memory/reembed":
            try:
                return self._json(200, MemoryAdmin(self._cfg()).reembed())
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/self/attribute":
            try:
                return self._json(200, SelfAdmin(self._cfg()).set_attribute(obj))
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/self/attribute/delete":
            try:
                return self._json(200, SelfAdmin(self._cfg()).delete_attribute(obj))
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/self/revert":            # drop adaptations over one core trait
            try:
                return self._json(200, SelfAdmin(self._cfg()).revert_to_core(obj))
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/self/state":
            try:
                return self._json(200, SelfAdmin(self._cfg()).set_state(obj))
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path == "/api/mode":
            mode = str(obj.get("mode", "normal"))
            if mode not in ("normal", "knowledge"):
                return self._json(400, {"error": "mode must be normal | knowledge"})
            ctx = obj.get("big_lm2_ctx")
            if ctx:                                  # adjust the 4090 big-LM context before restart
                try:
                    cj = json.loads(Path(self.config_path).read_text()) \
                        if Path(self.config_path).exists() else {}
                    cj.setdefault("big_lm2", {})["ctx_size"] = int(ctx)
                    _atomic_json(Path(self.config_path), cj)
                except Exception as e:
                    return self._json(500, {"error": f"could not set big_lm2 ctx: {e}"})
            ctrl = LOGS_DIR / "control"
            ctrl.mkdir(parents=True, exist_ok=True)
            (ctrl / "mode").write_text(mode + "\n")
            (ctrl / "__restart__.req").write_text(str(time.time()))   # monitor → detached full restart
            return self._json(200, {"ok": True, "mode": mode, "restarting": True})
        if path == "/api/restart":
            name = str(obj.get("service", "")).replace("/", "").replace("..", "")
            if not name or not (LOGS_DIR / f"{name}.log").exists():
                return self._json(404, {"error": "unknown service"})
            try:
                self._request_restart(name)
                return self._json(200, {"ok": True, "service": name})
            except Exception as e:
                return self._json(500, {"error": str(e)})
        if path.startswith("/api/profiles/"):
            return self._handle_profiles(path, obj)
        self._json(404, {"error": "not found"})

    # ── Profiles (memory + personalities bundles) ──────────────────────────────
    def _request_restart(self, name: str):
        (LOGS_DIR / "control").mkdir(parents=True, exist_ok=True)
        (LOGS_DIR / "control" / f"{name}.req").write_text(str(time.time()))

    def _handle_profiles(self, path: str, obj: dict):
        action = path[len("/api/profiles/"):]
        try:
            if action == "switch":
                CFGMOD.set_active_profile(obj.get("name", ""))
                # Restart the services that hold the memory DB / personas so the switch
                # takes effect; the config server itself re-reads config per request.
                for svc in ("cascade", "research"):
                    if (LOGS_DIR / f"{svc}.log").exists():
                        self._request_restart(svc)
                return self._json(200, {"ok": True, "active": CFGMOD.active_profile()})
            if action == "create":
                CFGMOD.create_profile(obj.get("name", ""))
                return self._json(200, {"ok": True})
            if action == "duplicate":
                CFGMOD.duplicate_profile(obj.get("from") or CFGMOD.active_profile(),
                                         obj.get("to", ""))
                return self._json(200, {"ok": True})
            if action == "delete":
                CFGMOD.delete_profile(obj.get("name", ""))
                return self._json(200, {"ok": True})
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        except Exception as e:
            return self._json(500, {"error": str(e)})
        return self._json(404, {"error": "unknown profile action"})

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.json")
    args = ap.parse_args()
    Handler.config_path = args.config
    cfg = CFGMOD.load_config(args.config)
    host, port = cfg["config_server"]["host"], cfg["config_server"]["port"]
    print(f"[config] editor on http://{host}:{port}  (editing {args.config})", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
