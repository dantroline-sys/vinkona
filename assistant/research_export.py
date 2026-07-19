"""
Research → knowledge-host hand-off.

Turns Vinkona's research hoard (the `documents` table — see memory.store_document) into a folder of
`<hash>.md` drops the standalone knowledge-host ingests into its own schema (chunk → embed → distill
→ cards → kb_ask).  Transport is the filesystem: a durable, append-only outbox the host mines on its
own cadence (research_loop_spec.md §1).

What is exported: only NON-PERSONAL world knowledge — documents whose `kind` is research/plan.  The
user's own crawled mail/files (`kind='crawl'`, and any legacy row whose topic matches a configured
crawl source) are NEVER exported, so nothing personal leaks into the general knowledge base.

Idempotent: one file per normalised question (filename = its hash).  A byte-identical file is left
untouched (a host no-op).  Incremental runs export only questions touched since a rowid watermark;
a FULL run re-writes every question — which also repairs any file that was deleted from the folder.
"""

import hashlib
import json
import os
import re
import time
import typing as tp

try:
    from safety import sanitize_external
except Exception:                       # importlib-loaded context without cwd on sys.path
    import importlib.util as _ilu
    from pathlib import Path as _Path
    _spec = _ilu.spec_from_file_location("safety", _Path(__file__).resolve().parent / "safety.py")
    _safety = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_safety)
    sanitize_external = _safety.sanitize_external

_WM_KEY = "research.export_watermark"
_WS = re.compile(r"\s+")


def _norm_question(q: str) -> str:
    """Normalise a question for hashing: collapse whitespace, casefold (so trivial re-phrasing of
    whitespace/case maps to one file — deeper paraphrase is the host's job to converge, §5)."""
    return _WS.sub(" ", (q or "").strip()).casefold()


def question_hash(q: str) -> str:
    return hashlib.sha1(_norm_question(q).encode("utf-8")).hexdigest()[:16]


def _doc_question(row: dict) -> str:
    """The question a document answers: the research topic, or the plan question when the topic is
    the generic 'plan' bucket (work_plan_questions stores the question in `title`)."""
    topic = (row.get("topic") or "").strip()
    if topic == "plan":
        return (row.get("title") or "").strip()
    return topic


def _exportable_where(crawl_sources: tp.Iterable[str]) -> tuple[str, list]:
    """WHERE fragment + params for export-eligible documents: non-personal (kind not
    'crawl') and whose topic is not a configured crawl source (catches legacy rows
    written before `kind` existed)."""
    deny = [s for s in (crawl_sources or []) if s]
    where = "COALESCE(kind,'research') <> 'crawl'"
    params: list = []
    if deny:
        where += " AND topic NOT IN (%s)" % ",".join("?" * len(deny))
        params += deny
    return where, params


def _fetch(db, cols: str, where: str, params: list) -> list[dict]:
    sql = f"SELECT {cols} FROM documents WHERE {where} ORDER BY rowid"
    try:
        return [dict(r) for r in db.execute(sql, params).fetchall()]
    except Exception:
        # A pre-card_hint DB (the MemoryStore migration adds the column; a bare test
        # fixture or an old snapshot may lack it) — export still works, just unhinted.
        return [dict(r) for r in db.execute(
            sql.replace(" card_hint,", ""), params).fetchall()]


_FULL_COLS = "rowid AS rowid, url, title, topic, text, digest, card_hint, fetched_at, kind"


def exportable_rows(db, crawl_sources: tp.Iterable[str]) -> list[dict]:
    """Every export-eligible document, full columns (kept for callers/tests; the
    incremental path below avoids pulling `text` for untouched questions)."""
    where, params = _exportable_where(crawl_sources)
    return _fetch(db, _FULL_COLS, where, params)


def render_doc(question: str, docs: list[dict], max_source_chars: int = 40000) -> str:
    """One `<hash>.md`: YAML front-matter (question + accumulated source ids) then the question,
    a short answer (cached digests, if any), and every source block.  Source text is sanitised
    (untrusted) and bounded so a single hoard row can't produce an unbounded file."""
    src_ids, sources, digests = [], [], []
    for d in docs:
        sid = (d.get("url") or "").strip() or (d.get("title") or "").strip() or f"doc{d.get('rowid')}"
        src_ids.append(sid)
        if (d.get("digest") or "").strip():
            digests.append(d["digest"].strip())
        body = sanitize_external((d.get("text") or "").strip(), max_source_chars)
        label = (d.get("title") or "").strip()
        url = (d.get("url") or "").strip()
        head = " · ".join(x for x in (label, url) if x) or "source"
        sources.append(f"### {head}\n\n{body}".strip())
    fm_sources = "\n".join(f"  - {s}" for s in src_ids)
    updated = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(
        max((d.get("fetched_at") or 0) for d in docs) or time.time()))
    q_yaml = question.replace('"', "'")
    # Card hint (brains): the newest doc carrying one wins.  It becomes two extra
    # front-matter scalars (card_type + one-line JSON context_features) that the host's
    # parser lifts, plus the shaped answer — which supersedes the generic digests as
    # `## Answer` (it IS the question-shaped synthesis, and the host runs its typed-card
    # extractor on that Answer chunk).
    hint = _pick_hint(docs)
    hint_lines = ""
    if hint.get("card_type"):
        hint_lines = f"card_type: {hint['card_type']}\n"
        if hint.get("context_features"):
            hint_lines += ("context_features: "
                           + json.dumps(hint["context_features"], ensure_ascii=False,
                                        separators=(",", ": ")) + "\n")
    if hint.get("answer"):
        digests = [hint["answer"]]
    # Front-matter + headings must match the host's parser exactly (knowledgehost/research.py):
    #   provenance: vinkona  (its is_research_doc gate) · trust: low · the question under `# Question`
    #   · `## Answer` (synthesis fallback) · `## Sources` with a `### <source>` per hoard row.
    parts = [f'---\nprovenance: vinkona\nkind: research\ntrust: low\n'
             f'hash: {question_hash(question)}\nupdated: {updated}\nquestion: "{q_yaml}"\n'
             f'{hint_lines}sources:\n{fm_sources}\n---\n',
             f"# Question\n\n{question}\n"]
    if digests:
        parts.append("## Answer\n\n" + "\n\n".join(digests) + "\n")
    parts.append("## Sources\n\n" + "\n\n".join(sources))
    return "\n".join(parts).strip() + "\n"


def _pick_hint(docs: list[dict]) -> dict:
    """The card hint to ship for this question: the NEWEST doc's non-empty, parseable
    card_hint (docs arrive rowid-ascending).  {} when no doc carries one."""
    for d in reversed(docs):
        raw = (d.get("card_hint") or "").strip()
        if not raw:
            continue
        try:
            hint = json.loads(raw)
        except ValueError:
            continue
        if isinstance(hint, dict):
            return hint
    return {}


def _hash16(content: str) -> str:
    """Drop-content fingerprint — a cross-repo contract with the knowledge
    host's research.drop_inventory (sha256[:16]); change both or neither."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0", ""}


def _is_local_url(url: str) -> bool:
    from urllib.parse import urlparse
    try:
        return (urlparse(url).hostname or "").lower() in _LOCAL_HOSTS
    except ValueError:
        return False


def resolve_export_target(cfg: dict) -> dict:
    """Where should research drops go so they are actually READ?

    The old contract made the operator route by hand (research.export.folder =
    path or URL) — and a path plus a REMOTE knowledge host quietly wrote drops
    into a local outbox nothing ever mines.  Resolution now follows the
    knowledge itself:

      folder is a URL                → http to it (explicit routing wins)
      transport pinned folder/http   → as pinned
      knowledge_host REMOTE          → http to it (a local outbox is a black
                                       hole; the configured folder is kept as
                                       a fallback for when the host is down)
      folder set (host local/absent) → the folder (the host mines it there)
      no folder, knowledge_host set  → http to it (its /drop writes the
                                       host's own solved dir — local or not)
      neither                        → off

    Returns {mode: http|folder|off, dest, token, reason[, fallback_folder]}.
    The token falls back to knowledge_host.token — same host, same secret."""
    rcfg = (cfg.get("research") or {}).get("export") or {}
    kh = cfg.get("knowledge_host") or {}
    folder = str(rcfg.get("folder") or "").strip()
    transport = str(rcfg.get("transport") or "auto").strip().lower()
    token = str(rcfg.get("token") or kh.get("token") or "")
    kh_url = str(kh.get("url") or "").strip() if kh.get("enabled") else ""

    if folder.startswith(("http://", "https://")):
        return {"mode": "http", "dest": folder, "token": token,
                "reason": "export folder is a URL"}
    if transport == "folder":
        if folder:
            return {"mode": "folder", "dest": os.path.expanduser(folder), "token": "",
                    "reason": "transport pinned to folder"}
        return {"mode": "off", "dest": "", "token": "",
                "reason": "transport pinned to folder but no folder configured"}
    if transport == "http":
        if kh_url:
            return {"mode": "http", "dest": kh_url, "token": token,
                    "reason": "transport pinned to http"}
        return {"mode": "off", "dest": "", "token": "",
                "reason": "transport pinned to http but no knowledge_host configured"}
    if kh_url and not _is_local_url(kh_url):
        out = {"mode": "http", "dest": kh_url, "token": token,
               "reason": "knowledge host is remote — a local outbox would never be read"}
        if folder:
            out["fallback_folder"] = os.path.expanduser(folder)
        return out
    if folder:
        return {"mode": "folder", "dest": os.path.expanduser(folder), "token": "",
                "reason": "local folder outbox"}
    if kh_url:
        return {"mode": "http", "dest": kh_url, "token": token,
                "reason": "no folder configured — delivering straight to the knowledge host"}
    return {"mode": "off", "dest": "", "token": "",
            "reason": "no export folder and no knowledge_host configured"}


def negotiate_drop(base_url: str, token: str, timeout: float = 10.0):
    """GET /drop — the server-server handshake.  Returns (status, payload):
      ("ok", {accepts, drops, ...})  the host answered — deliver accordingly
      ("denied", None)               reachable but the token was refused
      ("no-route", None)             reachable, but an older host without the
                                     handshake — POSTing blind still works
      ("down", None)                 unreachable — don't burn the outbox on it
    """
    import urllib.error
    import urllib.request
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(base_url.rstrip("/") + "/drop", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            res = json.loads(r.read().decode("utf-8", "replace"))
        return ("ok", res) if res.get("ok") else ("no-route", None)
    except urllib.error.HTTPError as e:
        return ("denied", None) if e.code in (401, 403) else ("no-route", None)
    except Exception:
        return ("down", None)


def run_export(memory, cfg: dict, crawl_sources: tp.Iterable[str] = (), *,
               full: bool = False) -> dict:
    """The smart lane: resolve where the knowledge host actually reads,
    handshake with it, then export only what it doesn't already hold.
    This is what the research worker calls; export_research stays the
    lower-level engine (and the fixed-destination API)."""
    target = resolve_export_target(cfg)
    rcfg = (cfg.get("research") or {}).get("export") or {}
    kh = cfg.get("knowledge_host") or {}
    kh_url = str(kh.get("url") or "").strip() if kh.get("enabled") else ""
    max_chars = int(rcfg.get("max_source_chars", 40000))
    base = {"transport": target["mode"], "dest": target["dest"],
            "route_reason": target["reason"]}

    def _gaps(hs) -> list:
        """The handshake's return leg: the host's open knowledge gaps as
        VERBATIM query strings (close_gap matches lower/trim on them, so the
        text must round-trip through the research queue untouched)."""
        out, seen = [], set()
        for g in (hs or {}).get("gaps") or []:
            q = str(g.get("query") if isinstance(g, dict) else g or "").strip()
            if q and q.lower() not in seen:
                seen.add(q.lower())
                out.append(q)
        return out

    if target["mode"] == "off":
        return {"ok": False, "error": target["reason"], "written": 0, "skipped": 0, **base}
    if target["mode"] == "folder":
        res = export_research(memory, target["dest"], crawl_sources, full=full,
                              max_source_chars=max_chars)
        if kh_url:      # folder transport, but the host still answers the handshake
            status, hs = negotiate_drop(kh_url, target["token"] or str(kh.get("token") or ""))
            if status == "ok":
                res["gaps"] = _gaps(hs)
        return {**res, **base}
    status, hs = negotiate_drop(target["dest"], target["token"])
    if status == "denied":
        return {"ok": False, "error": f"knowledge host at {target['dest']} refused the "
                "token — check research.export.token / knowledge_host.token",
                "written": 0, "skipped": 0, **base}
    if status == "ok" and not hs.get("accepts", True):
        return {"ok": False, "error": f"knowledge host can't store drops: "
                f"{hs.get('reason') or 'no reason given'}",
                "written": 0, "skipped": 0, **base}
    if status == "down":
        if target.get("fallback_folder"):
            res = export_research(memory, target["fallback_folder"], crawl_sources,
                                  full=full, max_source_chars=max_chars)
            return {**res, **base, "transport": "folder",
                    "route_reason": "host unreachable — fell back to the folder outbox"}
        return {"ok": False, "error": f"knowledge host at {target['dest']} is "
                "unreachable — will retry next cycle", "written": 0, "skipped": 0, **base}
    inventory = (hs or {}).get("drops") if status == "ok" else None
    res = export_research(memory, target["dest"], crawl_sources, full=full,
                          max_source_chars=max_chars, token=target["token"],
                          inventory=inventory)
    if status == "ok":
        res["gaps"] = _gaps(hs)
    return {**res, **base}


def _post_drop(base_url: str, token: str, name: str, content: str,
               timeout: float = 30.0) -> bool:
    """POST one drop to a remote knowledge host's /drop route (Vinur on another
    machine).  The host byte-compares and answers {ok, changed} — same no-op
    semantics as _write_if_changed.  Returns True if the host wrote it."""
    import urllib.request
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(base_url.rstrip("/") + "/drop",
                                 data=json.dumps({"name": name, "content": content},
                                                 ensure_ascii=False).encode("utf-8"),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.loads(r.read().decode("utf-8", "replace"))
    if not res.get("ok"):
        raise RuntimeError(str(res.get("error") or "drop rejected"))
    return bool(res.get("changed"))


def _write_if_changed(path: str, content: str) -> bool:
    """Write only when the file is absent or differs — a byte-identical drop is a host no-op.
    Returns True if written."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            if f.read() == content:
                return False
    except FileNotFoundError:
        pass
    except Exception:
        pass                                          # unreadable → overwrite
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)                             # atomic
    return True


def export_research(memory, folder: str, crawl_sources: tp.Iterable[str] = (), *,
                    full: bool = False, max_source_chars: int = 40000,
                    token: str = "", inventory: tp.Optional[dict] = None) -> dict:
    """Write the research hoard out as `<hash>.md` drops under `folder`.

    `folder` may also be a remote knowledge host's base URL ("http://box:8771"):
    drops then POST to its /drop route (Bearer `token`) instead of the filesystem —
    same idempotent no-op on byte-identical content, host-side.

    Incremental (full=False): only questions with a document newer than the saved rowid watermark.
    Full (full=True): every question — repairs files deleted from the folder and re-syncs content.
    Byte-identical files are skipped.  Returns a summary dict for the trace/UI."""
    folder = (folder or "").strip()
    remote = folder.startswith(("http://", "https://"))
    if not remote:
        folder = os.path.expanduser(folder)
    if not folder:
        return {"ok": False, "error": "no export folder configured", "written": 0, "skipped": 0}
    if not remote:
        os.makedirs(folder, exist_ok=True)
    # Light scan first (no `text`): group every eligible doc by question, then pull the
    # heavy full rows only for questions actually touched this run — the hourly
    # incremental would otherwise load the whole lifetime hoard for a no-op.
    where, params = _exportable_where(crawl_sources)
    light = _fetch(memory.db, "rowid AS rowid, title, topic", where, params)

    groups: dict[str, dict] = {}
    for r in light:
        q = _doc_question(r)
        if not q:
            continue
        g = groups.setdefault(question_hash(q), {"question": q, "rowids": [], "max_rowid": 0})
        g["rowids"].append(r["rowid"])
        g["max_rowid"] = max(g["max_rowid"], r["rowid"])

    try:
        wm = 0 if full else int(memory.get_state(_WM_KEY) or 0)
    except Exception:
        wm = 0
    written = skipped = 0
    for h, g in groups.items():
        if not full and g["max_rowid"] <= wm:
            continue
        ids = ",".join(str(i) for i in g["rowids"])
        docs = _fetch(memory.db, _FULL_COLS, f"rowid IN ({ids})", [])
        content = render_doc(g["question"], docs, max_source_chars=max_source_chars)
        if remote:
            # Handshake inventory (run_export): the host told us what it holds —
            # a matching fingerprint means this drop is already there, skip the
            # bytes entirely (the full re-export becomes near-free over the wire).
            if inventory is not None and inventory.get(h + ".md") == _hash16(content):
                skipped += 1
                continue
            try:
                changed = _post_drop(folder, token, h + ".md", content)
            except Exception as e:
                # Abort WITHOUT advancing the watermark — the next run retries
                # everything from here (byte-identical re-posts are host no-ops).
                return {"ok": False, "error": f"remote drop failed: {e}", "folder": folder,
                        "written": written, "skipped": skipped, "questions": len(groups),
                        "documents": len(light), "full": full}
        else:
            changed = _write_if_changed(os.path.join(folder, h + ".md"), content)
        if changed:
            written += 1
        else:
            skipped += 1
    new_wm = max((r["rowid"] for r in light), default=wm)
    try:
        memory.set_state(_WM_KEY, str(new_wm))
    except Exception:
        pass
    return {"ok": True, "folder": folder, "written": written, "skipped": skipped,
            "questions": len(groups), "documents": len(light), "watermark": new_wm, "full": full}
