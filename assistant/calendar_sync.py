"""
calendar_sync.py — idle consolidation of the user's calendars into Vinkona's own.

At idle Vinkona reads every appointment across the user's calendars and mirrors each one into
the single "Vinkona" calendar (the user's phone-visible schedule), annotated with a short note
in her own voice.  A durable local copy (the calendar_events table) lets the voice model
answer "what's on today?" instantly, with no tool round-trip and even straight after a
restart.

Design constraints baked in here:
  • Read-broad, write-own.  Vinkona only ever writes to her OWN calendar; the originals on the
    user's other calendars are never touched.  The mirror is strictly one-way.
  • Loop-safe & idempotent.  Each mirror carries the origin event's UID in its notes (a marker
    the next pass parses back), so re-runs UPDATE in place rather than duplicating, and a
    content hash skips no-op writes.  The Vinkona calendar itself is the source of truth for
    "which mirrors exist", so the sync self-heals even if the local cache is wiped.
  • Vinkona never deletes an event she didn't create — pruning only ever removes her own tagged
    mirrors whose origin has disappeared (a cancelled appointment).

The reconcile logic (classify / plan_actions) is pure and deterministic so it is fully
unit-testable; all I/O (tool-host calls, big-LM notes) lives in the research worker
(sync_calendar) and the live scan (fold_mirrors) lives in the cascade.

Tool-host contract (Mac side — see the dev note): `calendar_range` must return events with
`id` (stable per occurrence), `calendar` (the source calendar's name), `title`, `start`,
`end`, `location`, `notes`; `calendar_create`/`calendar_update` must accept `notes` (so the
marker survives) and write to the Vinkona calendar, with `calendar_update` taking the mirror's
`id`.
"""

import datetime
import hashlib
import re
import time
import typing as tp

# Marker embedded in every mirror's notes: it ties the mirror back to its origin UID so the
# next pass recognises its own work.  The inner [vinkona-mirror:UID] is parsed; the whole line
# is stripped when recovering Vinkona's own note text.
#
# LEGACY: the assistant used to be called Amiga, and live calendars still carry her old
# markers ("kept in sync by Amiga [amiga-mirror:UID]").  Reads accept BOTH forms forever —
# otherwise every pre-rename mirror turns invisible: the sync duplicates instead of updating,
# prune can never claim an orphan, and the stale marker text leaks into visible notes.
# Writes only ever use the new template, and plan_actions upgrades a legacy-marked mirror in
# place on its next pass, so the old name disappears from the calendar by itself.
_MARKER_TEMPLATE = "\n\n— kept in sync by Vinkona [vinkona-mirror:{uid}] —"
_MARKER_INNER = re.compile(r"\[(?:vinkona|amiga)-mirror:([^\]]+)\]", re.IGNORECASE)
_MARKER_LINE = re.compile(
    r"\n*—\s*kept in sync by (?:Vinkona|Amiga)\s*\[(?:vinkona|amiga)-mirror:[^\]]+\]\s*—\s*$",
    re.IGNORECASE)
_LEGACY_MARKER = re.compile(r"\[amiga-mirror:", re.IGNORECASE)


# ── pure helpers ──────────────────────────────────────────────────────────────

def _field_str(v: tp.Any) -> str:
    """Coerce a raw event field to a plain string — NEVER raising on an odd type.  Unwraps the
    structured date objects some backends emit ({"dateTime": …} / {"date": …} for all-day), so a
    calendar that returns objects instead of ISO strings can't crash the reconcile."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for k in ("dateTime", "date", "value", "start", "text"):
            inner = v.get(k)
            if isinstance(inner, str):
                return inner
        return ""
    return str(v)


def normalize(ev: dict) -> dict:
    """Coerce one raw tool-host event into the field names the reconcile uses, tolerating the
    common synonyms different calendar backends emit AND any non-string / structured field shape
    (every value goes through _field_str, so classify can never throw on a weird event)."""
    if not isinstance(ev, dict):
        ev = {}
    title = _field_str(ev.get("title") or ev.get("summary")).strip()
    start = _field_str(ev.get("start") or ev.get("when")).strip()
    uid = ev.get("id") or ev.get("uid")
    return {
        "uid": str(uid) if uid else f"{title}@{start}",
        "title": title,
        "start": start,
        "end": _field_str(ev.get("end")).strip(),
        "location": _field_str(ev.get("location")).strip(),
        "calendar": _field_str(ev.get("calendar") or ev.get("cal")).strip(),
        "notes": _field_str(ev.get("notes") or ev.get("description")),
    }


def origin_uid_from_notes(notes: tp.Any) -> tp.Optional[str]:
    """The origin UID a mirror's notes carry, or None if this isn't one of Vinkona's mirrors."""
    notes = _field_str(notes)
    if not notes:
        return None
    m = _MARKER_INNER.search(notes)
    return m.group(1) if m else None


def comment_from_notes(notes: tp.Any) -> str:
    """Recover Vinkona's own note from a mirror's notes (strip the trailing marker line)."""
    notes = _field_str(notes)
    if not notes:
        return ""
    return _MARKER_LINE.sub("", notes).strip()


def build_notes(comment: str, uid: str) -> str:
    """Compose the notes field written to a mirror: Vinkona's comment + the origin marker."""
    marker = _MARKER_TEMPLATE.format(uid=uid)
    comment = (comment or "").strip()
    return (comment + marker) if comment else marker.lstrip("\n")


def _norm_txt(v: tp.Any) -> str:
    """Whitespace-collapse + case-fold a text field so trivial re-formatting isn't seen as a change."""
    return " ".join(_field_str(v).split()).casefold()


def event_hash(ev: dict) -> str:
    """Content identity for change detection — deliberately tolerant so that a re-export which only
    reformats the SAME appointment doesn't look changed: start/end are normalised to an instant (ISO
    jitter — trailing seconds, 'Z' vs +00:00 — collapses away), title/location are whitespace/
    case-folded, and Vinkona's own note is EXCLUDED (regenerating a note isn't an appointment change)."""
    def _t(k: str) -> str:
        e = to_epoch(ev.get(k))
        return f"{e:.0f}" if e is not None else _field_str(ev.get(k)).strip()
    key = "|".join((_norm_txt(ev.get("title")), _t("start"), _t("end"), _norm_txt(ev.get("location"))))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def match_key(ev: dict) -> tuple:
    """Identity for ADOPTING a pre-existing unmarked copy as a mirror: title + start only
    (location/notes may not have been copied verbatim).  Loose on purpose — better to adopt
    an existing copy than create a second one beside it."""
    return ((ev.get("title") or "").strip().lower(),
            to_epoch(ev.get("start")) or (ev.get("start") or ""))


def to_epoch(s: tp.Any) -> tp.Optional[float]:
    """Parse an ISO-8601 start/end into a unix timestamp; None if unparseable."""
    if not s:
        return None
    s = str(s).strip()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def _own_calendar_names(vinkona_calendar) -> set:
    """The set of calendar names treated as Vinkona's OWN — the configured name plus any
    legacy aliases (a string or a list both work).  Case-folded for the comparison."""
    names = (vinkona_calendar if isinstance(vinkona_calendar, (list, tuple, set))
             else [vinkona_calendar])
    return {str(n).strip().lower() for n in names if str(n or "").strip()}


def classify(events: list, vinkona_calendar="") -> tp.Tuple[list, dict, list]:
    """Split a flat event list into (origins, mirrors, own).

    origins — real appointments from calendars OTHER than Vinkona's, with no marker.
    mirrors — events carrying Vinkona's marker, keyed by the origin UID they represent (each
              carrying its own content hash and the mirror's calendar id, for the plan).
    own     — un-marked events sitting in the Vinkona calendar: Vinkona's OWN additions (booked
              from conversation), or anything the user put there directly.  These are NEVER
              written or pruned by the sync — but they ARE recorded (flagged self-authored)
              so Vinkona knows her own allocations apart from the mirrored ones.

    `vinkona_calendar` may be one name or a list — the pre-rename "Amiga" calendar must count
    as her own, or everything in it is misread as foreign origins and re-mirrored (duplicates).
    """
    own_names = _own_calendar_names(vinkona_calendar)
    origins: list = []
    mirrors: dict = {}
    own: list = []
    for raw in events:
        ev = normalize(raw)
        muid = origin_uid_from_notes(ev["notes"])
        if muid:
            mirrors[muid] = {**ev, "origin_uid": muid,
                             "vinkona_id": str(raw.get("id") or ev["uid"]),
                             "hash": event_hash(ev),
                             "note": comment_from_notes(ev["notes"]),
                             "legacy_marker": bool(_LEGACY_MARKER.search(ev["notes"]))}
        elif own_names and ev["calendar"].lower() in own_names:
            own.append({**ev, "vinkona_id": str(raw.get("id") or ev["uid"])})
        else:
            origins.append(ev)
    return origins, mirrors, own


def plan_actions(origins: list, mirrors: dict, own: tp.Optional[list] = None,
                 prune: bool = True) -> list:
    """Diff origins against existing mirrors → the create/update/skip/delete/adopt plan.

    Pure and deterministic.  Mirror↔origin is matched first by the origin UID and, when that
    fails, by a content signature (title+start) — some feeds (Hosportal) regenerate a fresh UID on
    every export, so a UID-only match would see every appointment as brand-new and delete-then-
    recreate the whole mirrored calendar each pull.  The content fallback recognises the churned
    event as the same one, so an UNCHANGED appointment SKIPs (no write) and a genuinely moved one
    UPDATEs in place.  Change detection compares content hashes (immune to time-format jitter);
    deletes only ever target Vinkona's own mirrors whose origin has truly vanished.

    `own` (optional): un-marked Vinkona-calendar entries.  When an origin has no mirror yet but
    matches an existing unmarked copy (title+start), emit ADOPT instead of CREATE — tag that
    copy as the mirror in place rather than duplicating it (the migration path for calendars
    that were previously copied verbatim without markers).
    """
    actions: list = []
    seen: set = set()                               # mirror keys (origin_uid) with a live origin
    own_by_key: dict = {}
    for o in (own or []):
        own_by_key.setdefault(match_key(o), o)      # first wins if the copy itself is duplicated
    mirror_by_key: dict = {}                         # content signature → mirror key, for UID churn
    for muid, m in mirrors.items():
        mirror_by_key.setdefault(match_key(m), muid)
    for ev in origins:
        uid = ev["uid"]
        h = event_hash(ev)
        m = mirrors.get(uid)
        mkey = uid if m is not None else None
        if m is None:                               # UID didn't match — try the content signature
            cand_uid = mirror_by_key.get(match_key(ev))
            if cand_uid is not None and cand_uid not in seen:
                m, mkey = mirrors[cand_uid], cand_uid
        if m is None:                               # genuinely new to the mirror
            cand = own_by_key.pop(match_key(ev), None)
            if cand is not None:                    # adopt the existing unmarked copy
                actions.append({"op": "adopt", "uid": uid, "event": ev,
                                "vinkona_id": cand["vinkona_id"], "hash": h})
            else:
                actions.append({"op": "create", "uid": uid, "event": ev, "hash": h})
            continue
        seen.add(mkey)                              # this mirror has a live origin → don't prune it
        if m.get("hash") != h or m.get("legacy_marker"):
            # legacy_marker: content may be unchanged, but the notes still carry the
            # pre-rename "Amiga" marker — update once to rewrite them in the new form
            # (the comment survives; the old name disappears from the visible calendar).
            actions.append({"op": "update", "uid": uid, "event": ev,
                            "vinkona_id": m["vinkona_id"], "hash": h, "note": m.get("note", "")})
        else:
            actions.append({"op": "skip", "uid": uid, "event": ev,
                            "vinkona_id": m["vinkona_id"], "hash": h, "note": m.get("note", "")})
    # Prune only when we actually saw origins.  Zero origins alongside existing mirrors is the
    # fingerprint of a FAILED / empty external read (network drop, timeout) — NOT a mass
    # cancellation.  Pruning then would wipe the entire mirrored calendar off a transient error,
    # so we never delete on a suspect-empty read (a genuinely emptied calendar just keeps its
    # stale mirrors one more cycle, until a good read confirms the removal).
    if prune and origins:
        for uid, m in mirrors.items():
            if uid not in seen:
                actions.append({"op": "delete", "uid": uid, "vinkona_id": m["vinkona_id"]})
    return actions


def fold_mirrors(events: list, vinkona_calendar: str = "") -> list:
    """Collapse origin events and their Vinkona-calendar mirrors to ONE entry per appointment
    (the mirror wins, carrying Vinkona's note), so the live scan over a consolidated calendar
    doesn't double-count.  A no-op when nothing is mirrored (no markers present).  Used by
    the cascade's calendar scan + proactive feed."""
    by_uid: dict = {}
    order: list = []
    for raw in events:
        ev = normalize(raw)
        muid = origin_uid_from_notes(ev["notes"])
        is_mirror = muid is not None
        uid = muid if is_mirror else ev["uid"]
        entry = {"uid": uid, "title": ev["title"], "start": ev["start"], "end": ev["end"],
                 "location": ev["location"],
                 "note": comment_from_notes(ev["notes"]) if is_mirror else "",
                 "_mirror": is_mirror}
        cur = by_uid.get(uid)
        if cur is None:
            by_uid[uid] = entry
            order.append(uid)
        elif is_mirror and not cur["_mirror"]:
            by_uid[uid] = entry          # the mirror (with its note) supersedes the bare origin
    return [by_uid[u] for u in order]


# ── durable local copy ─────────────────────────────────────────────────────────

class CalendarStore:
    """Durable local copy of Vinkona's consolidated schedule — the instant-retrieval cache.

    Refreshed wholesale each idle sync (one transaction) and queried synchronously by the
    voice path, so "what's on today?" needs no tool round-trip and survives a restart.
    Shares the MemoryStore connection."""

    def __init__(self, db):
        self.db = db
        self._init_db()

    def _init_db(self) -> None:
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS calendar_events (
            uid TEXT PRIMARY KEY, vinkona_id TEXT, title TEXT, start TEXT, "end" TEXT,
            start_ts REAL, location TEXT, source TEXT, note TEXT, hash TEXT, synced_at REAL);
        CREATE INDEX IF NOT EXISTS idx_calendar_start ON calendar_events(start_ts);
        """)
        # self_authored: 1 = Vinkona's own addition (booked from conversation), 0 = mirrored from
        # an external calendar.  Idempotent ALTER so older caches gain the column in place.
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(calendar_events)")}
        if "self_authored" not in cols:
            self.db.execute("ALTER TABLE calendar_events ADD COLUMN self_authored INTEGER DEFAULT 0")
        self.db.commit()

    def replace_all(self, rows: list) -> None:
        """Swap in a fresh snapshot of the consolidated schedule (atomic)."""
        with self.db:
            self.db.execute("DELETE FROM calendar_events")
            if rows:
                self.db.executemany(
                    'INSERT OR REPLACE INTO calendar_events '
                    '(uid,vinkona_id,title,start,"end",start_ts,location,source,note,hash,synced_at,'
                    'self_authored) VALUES (:uid,:vinkona_id,:title,:start,:end,:start_ts,:location,'
                    ':source,:note,:hash,:synced_at,:self_authored)',
                    [{"self_authored": 0, **r} for r in rows])   # default mirrored unless flagged

    def upcoming(self, now: tp.Optional[float] = None, horizon_days: tp.Optional[float] = None,
                 limit: int = 50) -> list:
        """Events from now (less a one-hour grace, so something on right now still shows)
        forward, soonest first."""
        now = time.time() if now is None else now
        sql = "SELECT * FROM calendar_events WHERE start_ts IS NOT NULL AND start_ts >= ?"
        params: list = [now - 3600]
        if horizon_days:
            sql += " AND start_ts <= ?"
            params.append(now + horizon_days * 86400)
        sql += " ORDER BY start_ts ASC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.db.execute(sql, params).fetchall()]

    def all(self) -> list:
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM calendar_events ORDER BY start_ts ASC").fetchall()]

    def summary(self, now: tp.Optional[float] = None, horizon_days: float = 1.0,
                max_items: int = 6) -> str:
        """A compact human line of what's coming up, for instant spoken retrieval."""
        evs = self.upcoming(now=now, horizon_days=horizon_days, limit=max_items)
        if not evs:
            return ""
        out = []
        for e in evs:
            when = ""
            if e.get("start_ts"):
                when = datetime.datetime.fromtimestamp(e["start_ts"]).strftime("%a %H:%M")
            line = f"{when} — {e['title']}".strip(" —")
            if e.get("self_authored"):
                line += " [you asked me to add this]"
            elif e.get("source"):
                line += f" [from your {e['source']} calendar]"
            if e.get("note"):
                line += f" ({e['note']})"
            out.append(line)
        return "\n".join(out)
