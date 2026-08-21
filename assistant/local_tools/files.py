"""Local files genre — file_search / file_list / file_read / file_index.

Scope is the configured `roots` list and NOTHING else: every path argument is
resolved and must land inside a root, so neither the model nor a crawler can
wander the machine.  Empty roots = an enabled-but-unconfigured genre; each
tool says so instead of pretending the disk is empty.  file_index is the
§4-compliant CRAWL lister (JSON-array-as-a-string, objects with `path`,
stable path-sorted order, honest offset/limit, "[]" past the end).
"""
import json
import os

from . import extract

WALK_CAP = 50_000            # entries visited per walk — a runaway tree stops here


def _roots(gcfg: dict) -> list:
    out = []
    for r in gcfg.get("roots") or []:
        p = os.path.realpath(os.path.expanduser(str(r).strip()))
        if p and os.path.isdir(p):
            out.append(p)
    return out


def _inside(path: str, roots: list) -> str | None:
    """The resolved path if it sits inside a configured root, else None."""
    p = os.path.realpath(os.path.expanduser(str(path or "").strip()))
    for r in roots:
        if p == r or p.startswith(r + os.sep):
            return p
    return None


def _walk_files(base: str) -> list:
    """All regular files under base, path-sorted (the stable crawl order),
    hidden entries and common noise dirs skipped, bounded by WALK_CAP."""
    found, seen = [], 0
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv"}
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and d not in skip_dirs)
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            seen += 1
            if seen > WALK_CAP:
                return found
            found.append(os.path.join(dirpath, name))
    return found


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n / 1.0:.1f}{unit}"
        n /= 1024
    return f"{n}B"


def tools(gcfg: dict, env: dict) -> list:
    roots = _roots(gcfg)
    max_read = int(gcfg.get("max_read_chars", 20000))

    def _need_roots():
        return {"ok": False, "error": "no folders are shared with the local file "
                "tools yet — add one or more under Settings → Local tools → Files"}

    def file_search(args):
        if not roots:
            return _need_roots()
        query = str(args.get("query") or "").strip().lower()
        if not query:
            return {"ok": False, "error": "give a search word or phrase"}
        limit = max(1, min(int(args.get("limit") or 5), 25))
        terms = query.split()
        hits = []
        for root in roots:
            for p in _walk_files(root):
                hay = os.path.relpath(p, root).lower()
                if all(t in hay for t in terms):
                    hits.append(p)
        if not hits:
            return f"No files matching '{query}' under the shared folders."
        lines = [f"Found {min(len(hits), limit)} of {len(hits)} matching file(s):"]
        for p in hits[:limit]:
            try:
                size = _fmt_size(os.path.getsize(p))
            except OSError:
                size = "?"
            lines.append(f"- {p} ({size})")
        return "\n".join(lines)

    def file_list(args):
        if not roots:
            return _need_roots()
        path = str(args.get("path") or "").strip()
        limit = max(1, min(int(args.get("limit") or 40), 200))
        if not path:
            return "Shared folders:\n" + "\n".join(f"- {r}" for r in roots)
        p = _inside(path, roots)
        if p is None or not os.path.isdir(p):
            return {"ok": False, "error": f"{path} is not inside a shared folder"}
        entries = sorted(os.listdir(p))
        shown = [e + ("/" if os.path.isdir(os.path.join(p, e)) else "")
                 for e in entries if not e.startswith(".")][:limit]
        return f"{p}:\n" + ("\n".join(f"- {e}" for e in shown) if shown else "(empty)")

    def file_read(args):
        if not roots:
            return _need_roots()
        p = _inside(args.get("path"), roots)
        if p is None:
            return {"ok": False, "error": f"{args.get('path')} is not inside a shared folder"}
        cap = min(int(args.get("max_chars") or max_read), max_read)
        ok, text = extract.extract_text(p, cap)
        return text if ok else {"ok": False, "error": text}

    def file_index(args):
        if not roots:
            return _need_roots()
        p = _inside(args.get("path"), roots)
        if p is None or not os.path.isdir(p):
            return {"ok": False, "error": f"{args.get('path')} is not inside a shared folder"}
        offset = max(0, int(args.get("offset") or 0))
        limit = max(1, min(int(args.get("limit") or 8), 100))
        page = _walk_files(p)[offset:offset + limit]
        items = []
        for f in page:
            try:
                size = os.path.getsize(f)
            except OSError:
                size = 0
            items.append({"path": f, "name": os.path.basename(f), "size": size})
        return json.dumps(items)

    hint = " Shared folders: " + ", ".join(roots) if roots else ""
    return [
        ({"name": "file_search",
          "description": "Search the user's shared folders for files by words in the "
                         "name or path; returns matching paths." + hint,
          "parameters": {"type": "object", "properties": {
              "query": {"type": "string", "description": "words to look for"},
              "limit": {"type": "integer", "default": 5}}, "required": ["query"]}},
         file_search),
        ({"name": "file_list",
          "description": "Browse a shared folder (no path lists the shared roots).",
          "parameters": {"type": "object", "properties": {
              "path": {"type": "string"}, "limit": {"type": "integer", "default": 40}}}},
         file_list),
        ({"name": "file_read",
          "description": "Read the text of one file in a shared folder "
                         "(text/code, docx, odt, xlsx, pptx, html; pdf if pypdf is installed).",
          "parameters": {"type": "object", "properties": {
              "path": {"type": "string"},
              "max_chars": {"type": "integer", "default": max_read}},
              "required": ["path"]}},
         file_read),
        ({"name": "file_index",
          "description": "CRAWL lister: files under a shared path as JSON, stable "
                         "path order, for offset paging. Separate from the interactive file_list.",
          "parameters": {"type": "object", "properties": {
              "path": {"type": "string"},
              "offset": {"type": "integer", "default": 0},
              "limit": {"type": "integer", "default": 8}}, "required": ["path"]}},
         file_index),
    ]


def probe(gcfg: dict, env: dict) -> dict:
    roots = _roots(gcfg)
    if not roots:
        raw = [str(r) for r in (gcfg.get("roots") or []) if str(r).strip()]
        if raw:
            return {"ok": False, "detail": "none of the configured folders exist "
                    "on this machine: " + ", ".join(raw)}
        return {"ok": False, "detail": "no folders configured yet — add the folders "
                "she may read (e.g. ~/Documents)"}
    counts = [f"{r} ({len(_walk_files(r))} files)" for r in roots]
    return {"ok": True, "detail": "readable: " + "; ".join(counts)}
