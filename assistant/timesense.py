"""
Vinkona's time-sense — Phase 1: the SEMANTIC CLOCK.

The wall clock tells the LM *what time it is*; this turns that into *what it means* —
part of day, weekend vs work-week, season, whether it's light or dark out, a public
holiday.  All computed locally and deterministically.  No learning here (usage rhythms /
recurrence detection are later phases); just the meaning the LM otherwise has to guess.

The deterministic core (part-of-day / weekend / season) needs no dependencies.  Two
enrichments are used only when available, and degrade silently otherwise:
  • astral   — sunrise/sunset, so "it's dark out" / "the sun's setting".  Needs
               coordinates: explicit lat/lon from config, else resolved from the location
               name via astral's small built-in geocoder.
  • holidays — names today's public holiday for a configured country (e.g. "GB").

Everything is best-effort: a missing library, an unknown city, or a bad value yields a
plainer line, never an error.
"""

import collections
import datetime
import functools
import importlib.util as _ilu
import statistics
import time

_DATEUTIL = _ilu.find_spec("dateutil") is not None

_ASTRAL = _ilu.find_spec("astral") is not None
_HOLIDAYS = _ilu.find_spec("holidays") is not None

# Hour cutoffs → the name for that stretch of the day (upper bound exclusive).
_PARTS = ((5, "the small hours"), (8, "early morning"), (12, "morning"),
          (14, "midday"), (18, "afternoon"), (22, "evening"), (24, "night"))

_SEASON_N = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring",
             5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn",
             10: "autumn", 11: "autumn"}
_SEASON_FLIP = {"winter": "summer", "summer": "winter", "spring": "autumn", "autumn": "spring"}


def part_of_day(hour: int) -> str:
    for hi, name in _PARTS:
        if hour < hi:
            return name
    return "night"


def season(month: int, *, southern: bool = False) -> str:
    s = _SEASON_N[month]
    return _SEASON_FLIP[s] if southern else s


def day_descriptor(weekday: int) -> str:
    """weekday: Monday=0 … Sunday=6."""
    if weekday >= 5:
        return "the weekend"
    if weekday == 0:
        return "the start of the work week"
    if weekday == 4:
        return "the end of the work week"
    return "midweek"


@functools.lru_cache(maxsize=8)
def _observer(location, lat, lon):
    """An astral Observer + tz name for the given coords/place, or None.  Cached, since
    it's resolved every turn and a geocoder lookup is the only non-trivial bit."""
    if not _ASTRAL:
        return None
    try:
        from astral import Observer
        if lat is not None and lon is not None:
            return Observer(float(lat), float(lon)), None
        if location:
            from astral.geocoder import lookup, database
            li = lookup(str(location).split(",")[0].strip(), database())
            return li.observer, li.timezone
    except Exception:
        return None
    return None


def _sun_phrase(now: datetime.datetime, location, lat, lon) -> str:
    obs = _observer(location, lat, lon)
    if not obs:
        return ""
    observer, tzname = obs
    try:
        from astral.sun import sun
        tz = None
        if tzname:
            import zoneinfo
            tz = zoneinfo.ZoneInfo(tzname)
        s = sun(observer, date=now.date(), tzinfo=tz)
        nowt = now if now.tzinfo else now.replace(tzinfo=tz)   # compare like-for-like
        hm = lambda dt: dt.strftime("%H:%M")
        if nowt < s["dawn"]:
            return f" It's still dark — sunrise at {hm(s['sunrise'])}."
        if nowt > s["dusk"]:
            return f" It's dark out — the sun set at {hm(s['sunset'])}."
        if abs((nowt - s["sunrise"]).total_seconds()) < 1800:
            return " The sun's just coming up."
        if abs((nowt - s["sunset"]).total_seconds()) < 1800:
            return " The sun's going down."
        return f" It's light out — sunset around {hm(s['sunset'])}."
    except Exception:
        return ""


def _holiday_phrase(now: datetime.datetime, country) -> str:
    if not (_HOLIDAYS and country):
        return ""
    try:
        import holidays
        name = holidays.country_holidays(str(country)).get(now.date())
        return f" It's {name}." if name else ""
    except Exception:
        return ""


def semantic_clock(now: datetime.datetime | None = None, *, location=None,
                   lat=None, lon=None, country=None, southern: bool | None = None) -> str:
    """A one-line meaning for the current moment, e.g.
    'It's evening on Saturday — the weekend, autumn. It's dark out — the sun set at 19:42.'"""
    now = now or datetime.datetime.now()
    if southern is None:
        try:
            southern = lat is not None and float(lat) < 0     # infer hemisphere from latitude
        except (TypeError, ValueError):
            southern = False
    base = (f"It's {part_of_day(now.hour)} on {now:%A} — "
            f"{day_descriptor(now.weekday())}, {season(now.month, southern=southern)}.")
    return base + _sun_phrase(now, location, lat, lon) + _holiday_phrase(now, country)


# ── Phase 2: learned usage rhythm ────────────────────────────────────────────
# How a part-of-day reads in the standing "you tend to be around in …" clause.
_PART_PHRASE = {"the small hours": "the small hours", "early morning": "early mornings",
                "morning": "mornings", "midday": "around midday", "afternoon": "afternoons",
                "evening": "evenings", "night": "late at night"}


def _join(items: list) -> str:
    items = list(items)
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + " and " + items[-1]


class UsageLog:
    """Append-only log of *when* the user is active, and the histograms over it that make
    Vinkona's sense of time relational ('you tend to be around in the evenings', 'it's later
    than you usually talk to me').  Shares the MemoryStore's sqlite/WAL connection.

    The table is `(ts, kind, label)` — Phase 2 logs `kind='session'` with no label; Phase 3
    (recurrence inference) reuses the same table with labelled events.  Pure counting, so
    it's robust and needs no model; it just stays quiet until there's enough history."""

    MIN_EVENTS = 12                          # below this, no rhythm is asserted

    def __init__(self, db):
        self.db = db
        self._init_db()

    def _init_db(self) -> None:
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS usage_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, label TEXT);
        CREATE INDEX IF NOT EXISTS idx_usage_kind ON usage_events(kind, ts);
        """)
        self.db.commit()

    def log(self, kind: str = "session", label: str | None = None, ts: float | None = None) -> None:
        self.db.execute("INSERT INTO usage_events(ts,kind,label) VALUES (?,?,?)",
                        (ts if ts is not None else time.time(), kind, label))
        self.db.commit()

    def _timestamps(self, kind: str = "session", label: str | None = None) -> list:
        if label is None:
            return [r[0] for r in self.db.execute(
                "SELECT ts FROM usage_events WHERE kind=? ORDER BY ts", (kind,))]
        return [r[0] for r in self.db.execute(
            "SELECT ts FROM usage_events WHERE kind=? AND label=? ORDER BY ts", (kind, label))]

    def labels(self, min_count: int = 4) -> list:
        """Distinct (kind, label) pairs with at least `min_count` events — the candidates
        worth running recurrence detection over."""
        return [(r["kind"], r["label"]) for r in self.db.execute(
            "SELECT kind, label, COUNT(*) c FROM usage_events WHERE label IS NOT NULL "
            "GROUP BY kind, label HAVING c >= ?", (min_count,))]

    def histograms(self, kind: str = "session") -> dict:
        hours = [0] * 24
        wdays = [0] * 7
        n = 0
        for ts in self._timestamps(kind):
            dt = datetime.datetime.fromtimestamp(ts)
            hours[dt.hour] += 1
            wdays[dt.weekday()] += 1
            n += 1
        return {"n": n, "hours": hours, "wdays": wdays}

    def summary(self, now: datetime.datetime | None = None, kind: str = "session") -> str:
        """A short, gentle line about how *this* moment sits against the user's usual rhythm,
        or '' when there isn't enough history (or no discernible pattern)."""
        now = now or datetime.datetime.now()
        h = self.histograms(kind)
        if h["n"] < self.MIN_EVENTS:
            return ""
        hours = h["hours"]
        mx = max(hours)
        if mx <= 0:
            return ""
        peak = [hr for hr, c in enumerate(hours) if c >= max(2, 0.5 * mx)]
        parts = []
        for hr in peak:
            p = part_of_day(hr)
            if p not in parts:
                parts.append(p)
        if not parts or len(parts) >= 5:         # no clear rhythm — say nothing
            return ""

        clauses = [f"You tend to be around in {_join([_PART_PHRASE[p] for p in parts])}"]

        # Is *now* one of those usual times, or notably off it?
        cur = hours[now.hour]
        if cur <= max(1, 0.15 * mx) and now.hour not in peak:
            earliest, latest = min(peak), max(peak)
            # The small hours after an afternoon/evening peak read as "late", not "early",
            # even though the clock number is small (it's the tail of the same night).
            if now.hour > latest or (now.hour < 5 and latest >= 17):
                clauses.append("it's later than you usually talk to me")
            elif now.hour < earliest:
                clauses.append("it's earlier than usual for you")
            else:
                clauses.append("this is an unusual hour for us")

        # Weekend vs work-week balance, only when notable and relevant to today.
        wk = h["wdays"]
        wend, wday = (wk[5] + wk[6]) / 2.0, sum(wk[0:5]) / 5.0
        if now.weekday() >= 5 and wday > 0 and wend < 0.6 * wday:
            clauses.append("weekends are usually quieter")
        elif now.weekday() < 5 and wend > 0 and wday < 0.6 * wend:
            clauses.append("you're usually more around at weekends")

        out = clauses[0]
        if len(clauses) > 1:
            out += " — " + _join(clauses[1:])
        return out + "."


# ── Phase 3: recurrence inference ("every nth day / week") ───────────────────
_WD = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _nth_weekday(dt: datetime.datetime) -> tuple:
    """(nth, is_last) for dt's weekday within its month — e.g. the 2nd Tuesday, or the
    last Friday.  nth is 1-based; is_last True when no later same-weekday day exists."""
    nth = (dt.day - 1) // 7 + 1
    is_last = (dt + datetime.timedelta(days=7)).month != dt.month
    return nth, is_last


def _dominant(values):
    """(value, count, fraction) of the most common item."""
    c = collections.Counter(values)
    val, cnt = c.most_common(1)[0]
    return val, cnt, cnt / len(values)


def _next_occurrence(kind: str, params: dict, last: datetime.datetime,
                     now: datetime.datetime, period_days: float) -> float:
    """The next time this rhythm is due, on/after `now`.  Uses dateutil.rrule for calendar
    rules when available; a simple last+period fallback otherwise."""
    if _DATEUTIL:
        try:
            from dateutil.rrule import (rrule, WEEKLY, MONTHLY,
                                        MO, TU, WE, TH, FR, SA, SU)
            wds = [MO, TU, WE, TH, FR, SA, SU]
            ref = now - datetime.timedelta(seconds=1)              # so "due right now" counts
            if kind in ("weekly", "biweekly"):
                r = rrule(WEEKLY, interval=2 if kind == "biweekly" else 1,
                          byweekday=wds[params["weekday"]], dtstart=last)
            elif kind == "monthly-dom":
                r = rrule(MONTHLY, bymonthday=params["dom"], dtstart=last)
            elif kind == "monthly-nth":
                nth = -1 if params["is_last"] else params["nth"]
                r = rrule(MONTHLY, byweekday=wds[params["weekday"]](nth), dtstart=last)
            else:
                r = None
            if r is not None:
                nxt = r.after(ref, inc=True)
                if nxt:
                    return nxt.timestamp()
        except Exception:
            pass
    # fallback / interval: step from last by the period until we're at/after now
    nxt = last
    step = datetime.timedelta(days=max(1.0, period_days))
    while nxt < now:
        nxt += step
    return nxt.timestamp()


def detect_rhythm(timestamps, now: datetime.datetime | None = None, *, min_support: int = 4):
    """Infer a recurrence from a series of occurrence timestamps (one label/thing), or None.

    Tries calendar rules first (weekly → biweekly → monthly-by-date → monthly-by-nth-weekday),
    then a skip-tolerant interval ('every ~N days').  Returns a dict with kind, period_days,
    a human `detail`, confidence, support, last_seen and next_expected.  Deterministic; a
    hypothesis, not a fact (the caller confidence-gates and lets the user dismiss)."""
    now = now or datetime.datetime.now()
    dts = sorted(datetime.datetime.fromtimestamp(t) for t in timestamps)
    if len(dts) < min_support:
        return None
    by_day = {}
    for dt in dts:                                    # one occurrence per calendar day
        by_day.setdefault(dt.date(), dt)
    dts = [by_day[d] for d in sorted(by_day)]
    if len(dts) < min_support:
        return None
    last = dts[-1]
    span = (dts[-1].date() - dts[0].date()).days
    n = len(dts)
    sup = min(1.0, n / 8.0)                            # support factor for confidence

    def result(kind, period, detail, conf, params):
        return {"kind": kind, "period_days": float(period), "detail": detail,
                "confidence": round(min(1.0, conf), 3), "support": n,
                "last_seen": last.timestamp(),
                "next_expected": _next_occurrence(kind, params, last, now, period)}

    # — weekly / biweekly: a dominant weekday AND ~7/14-day spacing (a shared weekday alone
    #   is also true of 'the last Friday', which is monthly — so check the cadence too) —
    wd, _, wd_frac = _dominant([d.weekday() for d in dts])
    if wd_frac >= 0.7 and span >= 21:
        same = [d for d in dts if d.weekday() == wd]
        gaps = [(same[i + 1].date() - same[i].date()).days for i in range(len(same) - 1)]
        med = statistics.median(gaps) if gaps else 7
        if 5 <= med <= 9:
            return result("weekly", 7, f"{_WD[wd]}s", wd_frac * sup, {"weekday": wd})
        if 12 <= med <= 16:
            return result("biweekly", 14, f"every other {_WD[wd]}", wd_frac * sup,
                          {"weekday": wd})
        # larger spacing on a fixed weekday → it's a monthly pattern; fall through

    # — monthly by date-of-month —
    if span >= 60:
        dom, _, dom_frac = _dominant([d.day for d in dts])
        if dom_frac >= 0.7:
            return result("monthly-dom", 30, f"the {_ordinal(dom)} of the month",
                          dom_frac * sup, {"dom": dom})
        # — monthly by nth weekday ("the last Friday", "the 2nd Tuesday") —
        keys = []
        for d in dts:
            nth, is_last = _nth_weekday(d)
            keys.append((d.weekday(), "last") if is_last else (d.weekday(), nth))
        key, _, nth_frac = _dominant(keys)
        if nth_frac >= 0.7:
            kwd, which = key
            if which == "last":
                detail, params = f"the last {_WD[kwd]} of the month", \
                    {"weekday": kwd, "nth": 5, "is_last": True}
            else:
                detail, params = f"the {_ordinal(which)} {_WD[kwd]} of the month", \
                    {"weekday": kwd, "nth": which, "is_last": False}
            return result("monthly-nth", 30, detail, nth_frac * sup, params)

    # — skip-tolerant interval ('every ~N days') —
    gaps = [(dts[i + 1].date() - dts[i].date()).days for i in range(len(dts) - 1)]
    med = statistics.median(gaps)
    if med >= 2:
        tol = max(1.0, 0.25 * med)
        ok = sum(1 for g in gaps if abs(g - max(1, round(g / med)) * med) <= tol)
        consistency = ok / len(gaps)
        if consistency >= 0.6:
            return result("interval", med, f"every {int(round(med))} days or so",
                          consistency * sup, {})
    return None


class RhythmStore:
    """Detected recurrences, persisted and curatable.  One row per label (its best rhythm).
    Low-trust hypotheses: confidence-gated, and the user can dismiss one (status='dismissed',
    never re-asserted).  Shares the MemoryStore connection."""

    MIN_CONFIDENCE = 0.5

    def __init__(self, db):
        self.db = db
        self._init_db()

    def _init_db(self) -> None:
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS rhythms (
            label TEXT PRIMARY KEY, kind TEXT, period_days REAL, detail TEXT,
            confidence REAL, support INTEGER, last_seen REAL, next_expected REAL,
            status TEXT DEFAULT 'active', updated_at REAL);
        """)
        self.db.commit()

    def upsert(self, label: str, r: dict) -> None:
        """Store/refresh a label's rhythm — unless the user dismissed it (that sticks)."""
        row = self.db.execute("SELECT status FROM rhythms WHERE label=?", (label,)).fetchone()
        if row and row["status"] == "dismissed":
            return
        self.db.execute(
            "INSERT INTO rhythms(label,kind,period_days,detail,confidence,support,last_seen,"
            "next_expected,status,updated_at) VALUES (?,?,?,?,?,?,?,?,'active',?) "
            "ON CONFLICT(label) DO UPDATE SET kind=excluded.kind,period_days=excluded.period_days,"
            "detail=excluded.detail,confidence=excluded.confidence,support=excluded.support,"
            "last_seen=excluded.last_seen,next_expected=excluded.next_expected,"
            "status='active',updated_at=excluded.updated_at",
            (label, r["kind"], r["period_days"], r["detail"], r["confidence"], r["support"],
             r["last_seen"], r["next_expected"], time.time()))
        self.db.commit()

    def set_status(self, label: str, status: str) -> None:
        self.db.execute("UPDATE rhythms SET status=?, updated_at=? WHERE label=?",
                        (status, time.time(), label))
        self.db.commit()

    def list(self, active_only: bool = True) -> list:
        q = "SELECT * FROM rhythms"
        if active_only:
            q += " WHERE status='active'"
        q += " ORDER BY confidence DESC"
        return [dict(r) for r in self.db.execute(q)]

    def refresh(self, usage, now: datetime.datetime | None = None) -> int:
        """Re-detect rhythms from the usage log: the session contact-cadence (label
        'contact') plus every labelled event series.  Returns how many were stored."""
        now = now or datetime.datetime.now()
        stored = 0
        sess = detect_rhythm(usage._timestamps("session"), now)
        if sess and sess["confidence"] >= self.MIN_CONFIDENCE:
            self.upsert("contact", sess); stored += 1
        for kind, label in usage.labels():
            r = detect_rhythm(usage._timestamps(kind, label), now)
            if r and r["confidence"] >= self.MIN_CONFIDENCE:
                self.upsert(label, r); stored += 1
        return stored

    def relevant(self, now: datetime.datetime | None = None, *, window_h: float = 18.0) -> str:
        """A short, gentle line for any rhythm that's DUE around now or OVERDUE — the
        anticipation payoff ('you usually do the shop around now').  '' when nothing's due."""
        now = now or datetime.datetime.now()
        nowt = now.timestamp()
        bits = []
        for r in self.list(active_only=True):
            label, nxt, period = r["label"], r["next_expected"] or 0, r["period_days"] or 1
            due_in_h = (nxt - nowt) / 3600.0
            overdue_days = (nowt - nxt) / 86400.0
            if label == "contact":
                # they're talking to us now; only affirm the pattern if today matches
                if -window_h <= due_in_h <= window_h:
                    bits.append("they usually check in around now")
                continue
            if abs(due_in_h) <= window_h:
                bits.append(f"they usually {label} around now ({r['detail']})")
            elif overdue_days >= max(1.0, 0.5 * period):
                bits.append(f"it's been longer than usual since they last {label}")
        if not bits:
            return ""
        return "What's usually due about now (mention only if it fits naturally): " + \
               _join(bits[:2]) + "."

