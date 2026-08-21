"""Local calendar genre — CalDAV (iCloud / Google / Nextcloud / Radicale / …).

Reads span ALL calendars in the account (so conflicts are caught); writes go
ONLY to the calendar named in tools.local.calendar.vinkona_calendar — enforced
HERE, exactly as MAC_TOOLS.md demands of any host, and not configurable away:
create refuses when that calendar doesn't exist (make one in the calendar
account first), update/delete refuse any event outside its collection.

The stdlib carries the whole protocol: PROPFIND discovery (principal →
calendar-home → collections), REPORT calendar-query for windowed reads, PUT /
DELETE on the one writable collection — all over the injected fetch (the
amiga_net broker in live use, canned XML in tests).  dateutil (already a hard
dependency) expands RRULEs, so recurring events appear as their occurrences
inside the asked window.

The result shapes are the Mac host's: calendar_range_json feeds calendar_sync
([{id,calendar,title,start,end,location,notes}]), calendar_create returns
{"created": …, "conflicts": …} with verified:true after a server read-back,
and write-verb tool NAMES stay create/update/delete so the spoken-confirmation
gate keeps triggering.
"""
import base64
import datetime
import json
import re
import uuid
import xml.etree.ElementTree as ET

from dateutil import rrule as _rrule
from dateutil import parser as _dtparser

_NS = {"D": "DAV:", "C": "urn:ietf:params:xml:ns:caldav"}


# ── small time helpers ───────────────────────────────────────────────────────

def _tz(env):
    name = (env.get("user_tz") or "").strip()
    if name:
        try:
            import zoneinfo
            return zoneinfo.ZoneInfo(name)
        except Exception:
            pass
    return datetime.datetime.now().astimezone().tzinfo


def _aware(dt, tz):
    return dt.replace(tzinfo=tz) if dt.tzinfo is None else dt


def _parse_ics_dt(value: str, params: dict, tz):
    """One DTSTART/DTEND/EXDATE value → aware datetime (all-day → 00:00 local)."""
    value = value.strip()
    if params.get("VALUE") == "DATE" or re.fullmatch(r"\d{8}", value):
        d = datetime.datetime.strptime(value, "%Y%m%d")
        return d.replace(tzinfo=tz), True
    if value.endswith("Z"):
        d = datetime.datetime.strptime(value, "%Y%m%dT%H%M%SZ")
        return d.replace(tzinfo=datetime.timezone.utc), False
    d = datetime.datetime.strptime(value, "%Y%m%dT%H%M%S")
    tzid = params.get("TZID")
    if tzid:
        try:
            import zoneinfo
            return d.replace(tzinfo=zoneinfo.ZoneInfo(tzid)), False
        except Exception:
            pass
    return d.replace(tzinfo=tz), False


def _ics_escape(s: str) -> str:
    return (str(s or "").replace("\\", "\\\\").replace(";", "\\;")
            .replace(",", "\\,").replace("\r\n", "\\n").replace("\n", "\\n"))


def _ics_unescape(s: str) -> str:
    return (str(s or "").replace("\\n", "\n").replace("\\N", "\n")
            .replace("\\,", ",").replace("\\;", ";").replace("\\\\", "\\"))


def _unfold(text: str) -> list:
    lines, out = text.replace("\r\n", "\n").replace("\r", "\n").split("\n"), []
    for ln in lines:
        if ln[:1] in (" ", "\t") and out:
            out[-1] += ln[1:]
        elif ln:
            out.append(ln)
    return out


def parse_vevents(ics: str, tz) -> list:
    """VEVENTs in one ics document → [{uid,title,start,end,all_day,location,
    notes,rrule,exdates}] with aware datetimes (unexpanded)."""
    events, cur = [], None
    for ln in _unfold(ics):
        if ln == "BEGIN:VEVENT":
            cur = {"uid": "", "title": "", "location": "", "notes": "",
                   "rrule": "", "exdates": [], "start": None, "end": None,
                   "all_day": False, "duration": None}
            continue
        if ln == "END:VEVENT":
            if cur and cur["start"] is not None:
                if cur["end"] is None:
                    span = cur["duration"] or (
                        datetime.timedelta(days=1) if cur["all_day"]
                        else datetime.timedelta(hours=1))
                    cur["end"] = cur["start"] + span
                events.append(cur)
            cur = None
            continue
        if cur is None or ":" not in ln:
            continue
        head, value = ln.split(":", 1)
        parts = head.split(";")
        prop = parts[0].upper()
        params = {}
        for p in parts[1:]:
            if "=" in p:
                k, v = p.split("=", 1)
                params[k.upper()] = v
        try:
            if prop == "UID":
                cur["uid"] = value.strip()
            elif prop == "SUMMARY":
                cur["title"] = _ics_unescape(value)
            elif prop == "LOCATION":
                cur["location"] = _ics_unescape(value)
            elif prop == "DESCRIPTION":
                cur["notes"] = _ics_unescape(value)
            elif prop == "DTSTART":
                cur["start"], cur["all_day"] = _parse_ics_dt(value, params, tz)
            elif prop == "DTEND":
                cur["end"], _ = _parse_ics_dt(value, params, tz)
            elif prop == "DURATION":
                m = re.fullmatch(
                    r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?", value.strip())
                if m:
                    d, h, mi, s = (int(x or 0) for x in m.groups())
                    cur["duration"] = datetime.timedelta(days=d, hours=h, minutes=mi, seconds=s)
            elif prop == "RRULE":
                cur["rrule"] = value.strip()
            elif prop == "EXDATE":
                for v in value.split(","):
                    cur["exdates"].append(_parse_ics_dt(v, params, tz)[0])
        except Exception:
            continue                      # one malformed property never sinks the event
    return events


def expand(events: list, win_start, win_end) -> list:
    """Occurrences that overlap [win_start, win_end): plain events pass through,
    RRULE events expand via dateutil (EXDATEs removed)."""
    out = []
    for ev in events:
        span = ev["end"] - ev["start"]
        if not ev["rrule"]:
            if ev["start"] < win_end and ev["end"] > win_start:
                out.append(dict(ev))
            continue
        try:
            rule = _rrule.rrulestr(ev["rrule"], dtstart=ev["start"])
            exdates = {d.astimezone(datetime.timezone.utc).replace(microsecond=0)
                       for d in ev["exdates"]}
            for occ_start in rule.between(win_start - span, win_end, inc=True):
                if occ_start.astimezone(datetime.timezone.utc).replace(microsecond=0) in exdates:
                    continue
                occ = dict(ev)
                occ["start"], occ["end"] = occ_start, occ_start + span
                occ["occurrence"] = True
                if occ["start"] < win_end and occ["end"] > win_start:
                    out.append(occ)
        except Exception:
            if ev["start"] < win_end and ev["end"] > win_start:
                out.append(dict(ev))
    out.sort(key=lambda e: e["start"])
    return out


# ── the CalDAV client ────────────────────────────────────────────────────────

class CalDav:
    def __init__(self, gcfg: dict, env: dict):
        self.base = str(gcfg.get("caldav_url") or "").strip().rstrip("/")
        self.user = str(gcfg.get("user") or "")
        self.vinkona_name = str(gcfg.get("vinkona_calendar") or "Vinkona").strip()
        self._auth = base64.b64encode(
            f"{self.user}:{gcfg.get('password') or ''}".encode()).decode()
        self._fetch = env["fetch"]
        self._tzinfo = _tz(env)
        self._now = env.get("now")
        self._uuid = env.get("uuid") or (lambda: uuid.uuid4().hex)
        import urllib.parse
        u = urllib.parse.urlsplit(self.base)
        self._origin = f"{u.scheme}://{u.netloc}"
        self._calendars = None

    def _url(self, href: str) -> str:
        return href if href.startswith("http") else self._origin + href

    def _req(self, url, method, body=None, depth=None, ctype="application/xml; charset=utf-8"):
        headers = {"Authorization": f"Basic {self._auth}"}
        if depth is not None:
            headers["Depth"] = str(depth)
        if body is not None:
            headers["Content-Type"] = ctype
        status, data = self._fetch(self._url(url), method=method,
                                   data=body.encode() if isinstance(body, str) else body,
                                   headers=headers, timeout=20.0, purpose="local_caldav")
        return status, data

    def _propfind(self, url, props, depth):
        inner = "".join(f"<{p}/>" for p in props)
        body = ('<?xml version="1.0"?><D:propfind xmlns:D="DAV:" '
                'xmlns:C="urn:ietf:params:xml:ns:caldav">'
                f"<D:prop>{inner}</D:prop></D:propfind>")
        status, data = self._req(url, "PROPFIND", body, depth=depth)
        if status >= 400:
            raise RuntimeError(f"PROPFIND {url} → HTTP {status}")
        return ET.fromstring(data)

    @staticmethod
    def _text(el, path):
        found = el.find(path, _NS)
        return (found.text or "").strip() if found is not None and found.text else ""

    def calendars(self) -> list:
        """[{href, name, writable_target}] — every calendar collection in the
        account's home, found by the standard discovery chain."""
        if self._calendars is not None:
            return self._calendars
        root = self._propfind(self.base, ("D:current-user-principal",), 0)
        principal = ""
        for resp in root.findall("D:response", _NS):
            principal = self._text(resp, ".//D:current-user-principal/D:href") or principal
        home = ""
        if principal:
            root = self._propfind(principal, ("C:calendar-home-set",), 0)
            for resp in root.findall("D:response", _NS):
                home = self._text(resp, ".//C:calendar-home-set/D:href") or home
        root = self._propfind(home or self.base,
                              ("D:resourcetype", "D:displayname"), 1)
        cals = []
        for resp in root.findall("D:response", _NS):
            href = self._text(resp, "D:href")
            if resp.find(".//D:resourcetype/C:calendar", _NS) is None:
                continue
            name = self._text(resp, ".//D:displayname") or href.rstrip("/").rsplit("/", 1)[-1]
            cals.append({"href": href, "name": name})
        self._calendars = cals
        return cals

    def vinkona_calendar(self) -> dict | None:
        for c in self.calendars():
            if c["name"].strip().lower() == self.vinkona_name.lower():
                return c
        return None

    def _fmt_utc(self, dt) -> str:
        return dt.astimezone(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def events_between(self, win_start, win_end, hrefs=None) -> list:
        """Expanded occurrences across the given calendars (default: all),
        each carrying id (its object href), calendar (display name)."""
        body = ('<?xml version="1.0"?><C:calendar-query xmlns:D="DAV:" '
                'xmlns:C="urn:ietf:params:xml:ns:caldav">'
                "<D:prop><D:getetag/><C:calendar-data/></D:prop>"
                '<C:filter><C:comp-filter name="VCALENDAR">'
                '<C:comp-filter name="VEVENT">'
                f'<C:time-range start="{self._fmt_utc(win_start)}" '
                f'end="{self._fmt_utc(win_end)}"/>'
                "</C:comp-filter></C:comp-filter></C:filter></C:calendar-query>")
        out = []
        for cal in (hrefs if hrefs is not None else self.calendars()):
            status, data = self._req(cal["href"], "REPORT", body, depth=1)
            if status >= 400:
                continue
            try:
                root = ET.fromstring(data)
            except ET.ParseError:
                continue
            for resp in root.findall("D:response", _NS):
                href = self._text(resp, "D:href")
                ics = self._text(resp, ".//C:calendar-data")
                for ev in expand(parse_vevents(ics, self._tzinfo), win_start, win_end):
                    ev["id"] = href + (f"#{ev['start'].isoformat()}"
                                       if ev.get("occurrence") else "")
                    ev["calendar"] = cal["name"]
                    out.append(ev)
        out.sort(key=lambda e: e["start"])
        return out

    def put_event(self, cal_href: str, fields: dict, uid: str = "",
                  href: str = "") -> str:
        """PUT one VEVENT; returns its object href.  Times are written UTC."""
        uid = uid or f"vinkona-{self._uuid()}"
        href = href or f"{cal_href.rstrip('/')}/{uid}.ics"
        now = datetime.datetime.fromtimestamp(
            float(self._now()) if self._now else 0.0, tz=datetime.timezone.utc)
        lines = ["BEGIN:VCALENDAR", "VERSION:2.0",
                 "PRODID:-//vinkona//local_tools//EN", "BEGIN:VEVENT",
                 f"UID:{uid}", f"DTSTAMP:{now.strftime('%Y%m%dT%H%M%SZ')}",
                 f"DTSTART:{self._fmt_utc(fields['start'])}",
                 f"DTEND:{self._fmt_utc(fields['end'])}",
                 f"SUMMARY:{_ics_escape(fields.get('title'))}"]
        if fields.get("location"):
            lines.append(f"LOCATION:{_ics_escape(fields['location'])}")
        if fields.get("notes"):
            lines.append(f"DESCRIPTION:{_ics_escape(fields['notes'])}")
        lines += ["END:VEVENT", "END:VCALENDAR"]
        status, _ = self._req(href, "PUT", "\r\n".join(lines) + "\r\n",
                              ctype="text/calendar; charset=utf-8")
        if status >= 400:
            raise RuntimeError(f"PUT → HTTP {status}")
        return href

    def get_event(self, href: str):
        status, data = self._req(href, "GET")
        if status >= 400:
            return None
        evs = parse_vevents(data.decode("utf-8", errors="replace"), self._tzinfo)
        return evs[0] if evs else None

    def delete(self, href: str) -> bool:
        status, _ = self._req(href, "DELETE")
        return status < 400


# ── the genre ────────────────────────────────────────────────────────────────

def _window(args, tz, now_fn):
    now = datetime.datetime.fromtimestamp(float(now_fn()), tz=tz)
    start_arg, end_arg = args.get("start"), args.get("end")
    if start_arg:
        start = _aware(_dtparser.parse(str(start_arg)), tz)
        end = (_aware(_dtparser.parse(str(end_arg)), tz) if end_arg
               else start + datetime.timedelta(days=1))
        return start, end
    days = max(1, min(int(args.get("days") or 7), 62))
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + datetime.timedelta(days=days)


def _fmt_ev(ev) -> str:
    s, e = ev["start"], ev["end"]
    if ev.get("all_day"):
        when = s.strftime("%a %d %b") + " (all day)"
    else:
        when = f"{s.strftime('%a %d %b %H:%M')}–{e.strftime('%H:%M')}"
    loc = f" @ {ev['location']}" if ev.get("location") else ""
    return f"{when} [{ev.get('calendar', '')}] {ev.get('title') or '(untitled)'}{loc}"


def tools(gcfg: dict, env: dict) -> list:
    tz = _tz(env)
    now_fn = env.get("now")

    def _need_config():
        return {"ok": False, "error": "the calendar account isn't configured yet — "
                "add the CalDAV address, username and app password under "
                "Settings → Local tools → Calendar"}

    def _client():
        return CalDav(gcfg, env) if str(gcfg.get("caldav_url") or "").strip() else None

    def _range_events(args):
        c = _client()
        if c is None:
            return None, _need_config()
        start, end = _window(args, tz, now_fn)
        return (start, end, c, c.events_between(start, end)), None

    def calendar_today(args):
        got, bad = _range_events({"days": 1})
        if bad:
            return bad
        _, _, _, evs = got
        if not evs:
            return "Nothing on the calendar today."
        return "Today:\n" + "\n".join(f"- {_fmt_ev(e)}" for e in evs)

    def calendar_range(args):
        got, bad = _range_events(args)
        if bad:
            return bad
        start, end, _, evs = got
        head = f"{start.strftime('%a %d %b')} – {end.strftime('%a %d %b')}"
        if not evs:
            return f"Nothing on the calendar {head}."
        return f"{head}:\n" + "\n".join(f"- {_fmt_ev(e)}" for e in evs)

    def calendar_range_json(args):
        got, bad = _range_events(args)
        if bad:
            return bad
        _, _, _, evs = got
        return json.dumps([{
            "id": e["id"], "calendar": e.get("calendar", ""),
            "title": e.get("title", ""), "start": e["start"].isoformat(),
            "end": e["end"].isoformat(), "location": e.get("location", ""),
            "notes": e.get("notes", ""),
        } for e in evs])

    def _parse_when(args):
        start = _aware(_dtparser.parse(str(args["start"])), tz)
        if args.get("end"):
            end = _aware(_dtparser.parse(str(args["end"])), tz)
        else:
            end = start + datetime.timedelta(hours=1)
        if end <= start:
            raise ValueError("the end must come after the start")
        return start, end

    def calendar_create(args):
        c = _client()
        if c is None:
            return _need_config()
        if not str(args.get("title") or "").strip():
            return {"ok": False, "error": "the event needs a title"}
        start, end = _parse_when(args)
        target = c.vinkona_calendar()
        if target is None:
            return {"ok": False, "error": f"no calendar named "
                    f"'{c.vinkona_name}' in the account — create one there first "
                    "(Vinkona only ever writes to her own calendar)"}
        conflicts = [ev for ev in c.events_between(start, end)
                     if ev["start"] < end and ev["end"] > start]
        if conflicts and not args.get("force"):
            return json.dumps({"created": False,
                               "conflicts": [_fmt_ev(e) for e in conflicts[:6]]})
        href = c.put_event(target["href"], {
            "title": args["title"], "start": start, "end": end,
            "location": args.get("location", ""), "notes": args.get("notes", "")})
        back = c.get_event(href)
        when = (f"{start.strftime('%a %d %b %H:%M')}–{end.strftime('%H:%M')}")
        return json.dumps({"created": True, "verified": back is not None,
                           "id": href, "when": when})

    def _own_event(c, event_id):
        """The (collection, plain object href) — or a refusal if the id points
        anywhere but the Vinkona calendar.  The write-containment gate."""
        target = c.vinkona_calendar()
        if target is None:
            return None, {"ok": False, "error": f"no calendar named "
                          f"'{c.vinkona_name}' in the account"}
        href = str(event_id or "").split("#")[0].strip()
        if not href.startswith(target["href"].rstrip("/") + "/"):
            return None, {"ok": False, "error": "that event is not on the "
                          f"'{c.vinkona_name}' calendar — Vinkona only ever "
                          "changes her own calendar"}
        return (target, href), None

    def calendar_update(args):
        c = _client()
        if c is None:
            return _need_config()
        own, bad = _own_event(c, args.get("id"))
        if bad:
            return bad
        target, href = own
        existing = c.get_event(href)
        if existing is None:
            return {"ok": False, "error": "no such event on the Vinkona calendar"}
        fields = {"title": args.get("title") or existing["title"],
                  "location": args.get("location") or existing["location"],
                  "notes": args.get("notes") if args.get("notes") is not None
                  else existing["notes"],
                  "start": existing["start"], "end": existing["end"]}
        if args.get("start"):
            fields["start"] = _aware(_dtparser.parse(str(args["start"])), tz)
            fields["end"] = fields["start"] + (existing["end"] - existing["start"])
        if args.get("end"):
            fields["end"] = _aware(_dtparser.parse(str(args["end"])), tz)
        c.put_event(target["href"], fields, uid=existing["uid"], href=href)
        back = c.get_event(href)
        return json.dumps({"updated": True, "verified": back is not None, "id": href})

    def calendar_delete(args):
        c = _client()
        if c is None:
            return _need_config()
        own, bad = _own_event(c, args.get("id"))
        if bad:
            return bad
        _, href = own
        if not c.delete(href):
            return {"ok": False, "error": "the server refused the delete"}
        return json.dumps({"deleted": True, "id": href})

    vname = str(gcfg.get("vinkona_calendar") or "Vinkona").strip()
    return [
        ({"name": "calendar_today",
          "description": "The user's calendar events for today, across all calendars.",
          "parameters": {"type": "object", "properties": {}}},
         calendar_today),
        ({"name": "calendar_range",
          "description": "Calendar events for the coming days (days=N) or an explicit "
                         "start/end, across all calendars.",
          "parameters": {"type": "object", "properties": {
              "days": {"type": "integer", "default": 7},
              "start": {"type": "string", "description": "ISO date/time"},
              "end": {"type": "string"}}}},
         calendar_range),
        ({"name": "calendar_range_json",
          "description": "Sync export: the same window as JSON "
                         "[{id,calendar,title,start,end,location,notes}].",
          "parameters": {"type": "object", "properties": {
              "days": {"type": "integer", "default": 7},
              "start": {"type": "string"}, "end": {"type": "string"}}}},
         calendar_range_json),
        ({"name": "calendar_create",
          "description": f"Add an event to the user's '{vname}' calendar (the "
                         "assistant's own calendar). Writes ONLY to that calendar. "
                         "Checks for conflicts across all calendars first. Times are "
                         "ISO-8601 in the user's local timezone.",
          "parameters": {"type": "object", "properties": {
              "title": {"type": "string"},
              "start": {"type": "string", "description": "ISO-8601, e.g. 2026-08-23T15:00"},
              "end": {"type": "string", "description": "ISO-8601; default +1h"},
              "location": {"type": "string"}, "notes": {"type": "string"},
              "force": {"type": "boolean",
                        "description": "create even if it clashes (default false)"}},
              "required": ["title", "start"]}},
         calendar_create),
        ({"name": "calendar_update",
          "description": f"Change an event on the '{vname}' calendar by id "
                         "(other calendars are never touched).",
          "parameters": {"type": "object", "properties": {
              "id": {"type": "string"}, "title": {"type": "string"},
              "start": {"type": "string"}, "end": {"type": "string"},
              "location": {"type": "string"}, "notes": {"type": "string"}},
              "required": ["id"]}},
         calendar_update),
        ({"name": "calendar_delete",
          "description": f"Remove an event from the '{vname}' calendar by id "
                         "(other calendars are never touched).",
          "parameters": {"type": "object", "properties": {
              "id": {"type": "string"}}, "required": ["id"]}},
         calendar_delete),
    ]


def probe(gcfg: dict, env: dict) -> dict:
    if not str(gcfg.get("caldav_url") or "").strip():
        return {"ok": False, "detail": "no CalDAV address configured yet — e.g. "
                "https://caldav.icloud.com (with an app-specific password) or "
                "your Nextcloud/Radicale URL"}
    c = CalDav(gcfg, env)
    cals = c.calendars()
    if not cals:
        return {"ok": False, "detail": "signed in, but found no calendars — check "
                "the address points at the account (not one calendar)"}
    names = ", ".join(x["name"] for x in cals[:10])
    has_own = c.vinkona_calendar() is not None
    tail = (f" '{c.vinkona_name}' found — appointments can be written."
            if has_own else
            f" NOTE no calendar named '{c.vinkona_name}' yet — create one in the "
            "account so she has somewhere safe to write; reads work regardless.")
    return {"ok": True, "detail": f"{len(cals)} calendar(s): {names}.{tail}"}
