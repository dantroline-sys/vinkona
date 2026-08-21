"""Local mail genre — read-only IMAP: mail_list / mail_recent / mail_search /
mail_read.

Accounts come from tools.local.mail.accounts ({label, host, port, user,
password}); use an app password where the provider offers one.  This genre
never sends, never deletes, never flags — every mailbox is opened read-only
(EXAMINE, not SELECT), so even the read cannot mark a message seen.  IMAP is
the toolset's one non-HTTP lane: a direct TLS connection from imaplib to the
server the user configured (the managed egress.toml block records this so the
posture stays honest).

mail_list is the §4 CRAWL lister: JSON-array-as-a-string, oldest-first by UID
(stable; new mail lands at the end), honest offset/limit, "[]" past the end.
Its id format is the Mac host's — "label|folder|uid" — passed straight to
mail_read.
"""
import email
import email.header
import email.utils
import json
import re

_FOLDER_ALIASES = {"inbox": "INBOX", "sent": "Sent", "drafts": "Drafts",
                   "trash": "Trash", "archive": "Archive", "junk": "Junk",
                   "spam": "Junk"}
_LIST_MBOX = re.compile(r'\(([^)]*)\)\s+"([^"]*)"\s+"?([^"]+?)"?$')


def _accounts(gcfg: dict) -> list:
    out = []
    for a in gcfg.get("accounts") or []:
        if str(a.get("host") or "").strip() and str(a.get("user") or "").strip():
            out.append({"label": str(a.get("label") or a.get("user") or "mail").strip(),
                        "host": str(a["host"]).strip(),
                        "port": int(a.get("port") or 993),
                        "user": str(a["user"]).strip(),
                        "password": str(a.get("password") or "")})
    return out


def _default_factory(host: str, port: int):
    import imaplib
    return imaplib.IMAP4_SSL(host, port, timeout=15)


def _pick(accounts: list, label: str) -> dict | None:
    label = (label or "").strip().lower()
    if not label:
        return accounts[0] if accounts else None
    for a in accounts:
        if a["label"].lower() == label:
            return a
    return None


def _connect(acct: dict, factory):
    conn = factory(acct["host"], acct["port"])
    conn.login(acct["user"], acct["password"])
    return conn


def _resolve_folder(conn, folder: str) -> str:
    """Case/alias-tolerant mailbox resolution, like the Mac host's (§5)."""
    want = (folder or "inbox").strip()
    want = _FOLDER_ALIASES.get(want.lower(), want)
    try:
        typ, boxes = conn.list()
    except Exception:
        return want
    if typ != "OK":
        return want
    names = []
    for raw in boxes or []:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        m = _LIST_MBOX.search(raw or "")
        if m:
            names.append(m.group(3))
    for n in names:
        if n.lower() == want.lower():
            return n
    for n in names:                      # provider prefixes: INBOX.Sent, [Gmail]/Sent Mail
        if n.lower().endswith(want.lower()) or want.lower() in n.lower():
            return n
    return want


def _hdr(msg, name: str) -> str:
    raw = msg.get(name, "")
    try:
        parts = email.header.decode_header(raw)
        return "".join(p.decode(enc or "utf-8", errors="replace")
                       if isinstance(p, bytes) else p
                       for p, enc in parts).strip()
    except Exception:
        return str(raw).strip()


def _body_text(msg) -> str:
    """The message's readable text: prefer text/plain, fall back to stripped
    text/html.  Attachments are named, never decoded."""
    from . import extract
    plains, htmls, attachments = [], [], []
    parts = msg.walk() if msg.is_multipart() else [msg]
    for part in parts:
        ctype = part.get_content_type()
        if part.get_content_maintype() == "multipart":
            continue
        fname = part.get_filename()
        if fname:
            attachments.append(_hdr(part, "Content-Disposition") and fname or fname)
            continue
        try:
            payload = part.get_payload(decode=True) or b""
            charset = part.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
        except Exception:
            continue
        if ctype == "text/plain":
            plains.append(text)
        elif ctype == "text/html":
            htmls.append(extract._strip_html(text))
    body = "\n".join(plains).strip() or "\n".join(htmls).strip()
    if attachments:
        body += "\n\n[attachments: " + ", ".join(attachments[:8]) + "]"
    return body.strip()


def _uids(conn, folder: str, criteria: tuple = ("ALL",)) -> list:
    typ, data = conn.uid("search", None, *criteria)
    if typ != "OK" or not data or not data[0]:
        return []
    return sorted(int(u) for u in data[0].split())


def _fetch_headers(conn, uid: int):
    typ, data = conn.uid("fetch", str(uid),
                         "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])")
    if typ != "OK" or not data:
        return None
    for part in data:
        if isinstance(part, tuple) and len(part) >= 2:
            return email.message_from_bytes(part[1])
    return None


def tools(gcfg: dict, env: dict) -> list:
    accounts = _accounts(gcfg)
    factory = env.get("imap_factory") or _default_factory
    read_cap = int(gcfg.get("max_read_chars", 8000))

    def _need_account(label=""):
        if not accounts:
            return {"ok": False, "error": "no mail account configured yet — add an "
                    "IMAP account (use an app password) under Settings → Local tools → Mail"}
        return {"ok": False, "error": f"no mail account labelled '{label}' — "
                "configured: " + ", ".join(a["label"] for a in accounts)}

    def _with_conn(label, folder, fn):
        acct = _pick(accounts, label)
        if acct is None:
            return _need_account(label)
        conn = _connect(acct, factory)
        try:
            box = _resolve_folder(conn, folder)
            typ, _ = conn.select(f'"{box}"', readonly=True)
            if typ != "OK":
                return {"ok": False, "error": f"no folder named {folder!r} on "
                        f"{acct['label']}"}
            return fn(conn, acct, box)
        finally:
            try:
                conn.logout()
            except Exception:
                pass

    def mail_list(args):
        offset = max(0, int(args.get("offset") or 0))
        limit = max(1, min(int(args.get("limit") or 8), 50))

        def run(conn, acct, box):
            uids = _uids(conn, box)[offset:offset + limit]
            items = []
            for uid in uids:
                msg = _fetch_headers(conn, uid)
                if msg is None:
                    continue
                items.append({"id": f"{acct['label']}|{box}|{uid}",
                              "from": _hdr(msg, "From"),
                              "subject": _hdr(msg, "Subject"),
                              "date": _hdr(msg, "Date"), "snippet": ""})
            return json.dumps(items)
        return _with_conn(args.get("account"), args.get("folder"), run)

    def mail_recent(args):
        limit = max(1, min(int(args.get("limit") or 5), 20))

        def run(conn, acct, box):
            uids = _uids(conn, box)[-limit:][::-1]
            if not uids:
                return f"{box} on {acct['label']} is empty."
            lines = [f"Latest {len(uids)} in {box} ({acct['label']}):"]
            for uid in uids:
                msg = _fetch_headers(conn, uid)
                if msg:
                    lines.append(f"- {_hdr(msg, 'Subject') or '(no subject)'} — "
                                 f"{_hdr(msg, 'From')} ({_hdr(msg, 'Date')})")
            return "\n".join(lines)
        return _with_conn(args.get("account"), args.get("folder") or "inbox", run)

    def mail_search(args):
        query = str(args.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "give a search word or phrase"}
        limit = max(1, min(int(args.get("limit") or 8), 25))
        days = int(args.get("days") or 0)

        def run(conn, acct, box):
            crit = ["TEXT", f'"{query}"']
            if days > 0:
                import datetime
                since = (datetime.date.today() - datetime.timedelta(days=days))
                crit = ["SINCE", since.strftime("%d-%b-%Y")] + crit
            try:
                uids = _uids(conn, box, tuple(crit))
            except Exception:
                uids = _uids(conn, box, ("CHARSET", "UTF-8") + tuple(crit))
            uids = uids[-limit:][::-1]
            if not uids:
                return f"No mail matching '{query}' in {box}."
            lines = [f"{len(uids)} message(s) matching '{query}':"]
            for uid in uids:
                msg = _fetch_headers(conn, uid)
                if msg:
                    lines.append(f"- [{acct['label']}|{box}|{uid}] "
                                 f"{_hdr(msg, 'Subject') or '(no subject)'} — "
                                 f"{_hdr(msg, 'From')} ({_hdr(msg, 'Date')})")
            return "\n".join(lines)
        return _with_conn(args.get("account"), args.get("folder") or "inbox", run)

    def mail_read(args):
        mid = str(args.get("id") or "").strip()
        parts = mid.split("|")
        if len(parts) != 3 or not parts[2].isdigit():
            return {"ok": False, "error": "id must look like label|folder|uid "
                    "(as mail_list returns)"}
        label, folder, uid = parts

        def run(conn, acct, box):
            typ, data = conn.uid("fetch", uid, "(BODY.PEEK[])")
            raw = None
            if typ == "OK":
                for part in data or []:
                    if isinstance(part, tuple) and len(part) >= 2:
                        raw = part[1]
            if raw is None:
                return {"ok": False, "error": f"no message {uid} in {box}"}
            msg = email.message_from_bytes(raw)
            head = (f"From: {_hdr(msg, 'From')}\nTo: {_hdr(msg, 'To')}\n"
                    f"Subject: {_hdr(msg, 'Subject')}\nDate: {_hdr(msg, 'Date')}\n\n")
            return head + _body_text(msg)[:read_cap]
        return _with_conn(label, folder, run)

    labels = ", ".join(a["label"] for a in accounts) or "none configured yet"
    return [
        ({"name": "mail_list",
          "description": "CRAWL lister: a page of emails in a folder as JSON, oldest "
                         f"first, for offset paging. Accounts: {labels}.",
          "parameters": {"type": "object", "properties": {
              "folder": {"type": "string", "description": "e.g. inbox, sent"},
              "offset": {"type": "integer", "default": 0},
              "limit": {"type": "integer", "default": 8},
              "account": {"type": "string"}}, "required": ["folder"]}},
         mail_list),
        ({"name": "mail_recent",
          "description": f"What's lately in the user's inbox (accounts: {labels}).",
          "parameters": {"type": "object", "properties": {
              "limit": {"type": "integer", "default": 5},
              "account": {"type": "string"}, "folder": {"type": "string"}}}},
         mail_recent),
        ({"name": "mail_search",
          "description": "Search mail by words; optionally only the last N days.",
          "parameters": {"type": "object", "properties": {
              "query": {"type": "string"}, "limit": {"type": "integer", "default": 8},
              "days": {"type": "integer"}, "account": {"type": "string"}},
              "required": ["query"]}},
         mail_search),
        ({"name": "mail_read",
          "description": "Read one email's full text by id (label|folder|uid, from "
                         "mail_list or mail_search).",
          "parameters": {"type": "object", "properties": {
              "id": {"type": "string"}}, "required": ["id"]}},
         mail_read),
    ]


def probe(gcfg: dict, env: dict) -> dict:
    accounts = _accounts(gcfg)
    if not accounts:
        return {"ok": False, "detail": "no account configured yet — host, user and "
                "an app password are the minimum"}
    factory = env.get("imap_factory") or _default_factory
    parts, ok = [], True
    for acct in accounts:
        try:
            conn = _connect(acct, factory)
            try:
                typ, data = conn.select('"INBOX"', readonly=True)
                count = int(data[0]) if typ == "OK" and data and data[0] else 0
                parts.append(f"{acct['label']}: signed in, INBOX holds {count} message(s)")
            finally:
                try:
                    conn.logout()
                except Exception:
                    pass
        except Exception as e:
            ok = False
            parts.append(f"{acct['label']}: FAILED — {e}")
    return {"ok": ok, "detail": "; ".join(parts)}
