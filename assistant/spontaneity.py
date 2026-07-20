"""Things Vinkona is holding that she might bring up — the segue lane.

A conversationalist mentions what they've been reading, or what they've been
trying to find out, when the moment offers a way in.  Assistants don't, and the
reason isn't capability: a badly-chosen aside ("speaking of which, did you see
the news about…") is memorably worse than silence, so the safe default is to
say nothing.  The problem is SELECTION, and it is solved here in three parts:

  1. a POOL of things she actually holds — headlines she crawled, findings from
     research she finished, questions she's still chasing, weather worth
     remarking on (ordinary weather is already in the ambient block; repeating
     it is not conversation);
  2. a SEGUE test — she offers something only when it genuinely touches what
     was just said.  No topical connection, no offer.  There is deliberately no
     "the conversation went quiet so say something" path;
  3. a WATERMARK — `offers` records what she actually said and what happened
     next.  The existing proactive feed tells the planner "never raise the same
     thing twice" as a prompt instruction with nothing behind it; an instruction
     is not a memory.

Only the last step is judged by the LM: the block offers a candidate and
explicitly permits saying nothing, because "no way in presented itself" has to
stay a normal outcome or the whole thing becomes a headline reader.

Engagement is recorded, so this shapes itself: taken-up offers become
acted-on rows in the user model (the same evidence the trait reflection reads),
and the pass-over count is reported beside them so the record stays two-sided.
"""
from __future__ import annotations

import re
import sqlite3
import time

KINDS = ("news", "finding", "question", "weather")

# Weather is only worth raising when it is NOT the usual — the ambient block
# already carries today's temperature into every single turn.
_NOTABLE_WEATHER = re.compile(
    r"\b(warning|alert|storm|gale|thunder|snow|sleet|hail|frost|ice|icy|flood|"
    r"heatwave|heavy rain|torrential|fog|blizzard|gust)\w*\b", re.I)

_WORD = re.compile(r"[a-z0-9][a-z0-9'-]+")
_STOP = frozenset("""
about after again against all also am an and any are aren as at be because been
before being below between both but by can cant come could couldnt did didnt do
does doesnt doing dont down during each few for from further get got had hadnt
has hasnt have havent having he her here hers herself him himself his how i if
in into is isnt it its itself just know let like me more most much must my
myself no nor not now of off on once only or other ought our ours out over own
really said same say see she should shouldnt so some such than that the their
theirs them themselves then there these they thing things this those through to
too under until up very was wasnt we well were what when where which while who
whom why will with wont would wouldnt yeah yes yet you your yours yourself
""".split())

# Turns that carry no topic of their own: they end a thread rather than steer it.
_ACK = frozenset("""
ok okay k right sure yeah yep yup mm mmm mhm uh-huh aha ah oh cool nice great
lovely fine good fair indeed quite true agreed exactly precisely thanks ta
cheers wow huh hmm hm
""".split())


def words(text: str) -> set[str]:
    """Distinctive lowercase words — the unit both the segue test and the
    did-she-actually-say-it test are measured in."""
    return {w for w in _WORD.findall((text or "").lower())
            if w not in _STOP and len(w) > 2}


def is_acknowledgement(text: str) -> bool:
    """A turn made only of 'mm', 'right', 'okay' — no topic of its own."""
    toks = _WORD.findall((text or "").lower())
    return bool(toks) and all(t in _ACK for t in toks) and len(toks) <= 4


# ── the watermark ────────────────────────────────────────────────────────────

class OfferLog:
    """What she has actually brought up, and what came of it.

    Rows appear only when she SPOKE the thing — a candidate she was offered and
    passed over stays available, because putting it in front of her isn't the
    same as raising it with the user."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self._migrate()

    def _migrate(self) -> None:
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS offers (
              key TEXT PRIMARY KEY,
              kind TEXT, text TEXT, session_id TEXT,
              first_at REAL, last_at REAL, times INTEGER DEFAULT 0,
              outcome TEXT DEFAULT 'pending'      -- pending | engaged | passed
            )""")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_offers_at ON offers(last_at)")
        self.db.commit()

    def spoken_keys(self) -> set[str]:
        return {r["key"] for r in self.db.execute("SELECT key FROM offers").fetchall()}

    def last_at(self) -> float:
        row = self.db.execute("SELECT MAX(last_at) t FROM offers").fetchone()
        return float(row["t"] or 0.0) if row else 0.0

    def count_since(self, since: float) -> int:
        row = self.db.execute("SELECT COUNT(*) c FROM offers WHERE last_at >= ?",
                              (since,)).fetchone()
        return int(row["c"] if row else 0)

    def record(self, item: dict, session_id: str = "", now: float | None = None) -> None:
        now = now or time.time()
        self.db.execute(
            "INSERT INTO offers(key,kind,text,session_id,first_at,last_at,times,outcome) "
            "VALUES (?,?,?,?,?,?,1,'pending') "
            "ON CONFLICT(key) DO UPDATE SET last_at=excluded.last_at, times=times+1",
            (item["key"], item.get("kind", ""), (item.get("text") or "")[:400],
             session_id, now, now))
        self.db.commit()

    def pending(self, session_id: str = "", max_age_s: float = 1800.0,
                now: float | None = None) -> dict | None:
        """The most recent thing she raised that hasn't been judged yet."""
        now = now or time.time()
        row = self.db.execute(
            "SELECT * FROM offers WHERE outcome='pending' AND last_at >= ? "
            + ("AND session_id=? " if session_id else "")
            + "ORDER BY last_at DESC LIMIT 1",
            ((now - max_age_s, session_id) if session_id else (now - max_age_s,))).fetchone()
        return dict(row) if row else None

    def resolve(self, key: str, outcome: str) -> None:
        self.db.execute("UPDATE offers SET outcome=? WHERE key=?", (outcome, key))
        self.db.commit()

    def summary(self, since: float = 0.0) -> str:
        """One even-handed line for the trait reflection: raised, taken up, and
        — the half that a positive-only record would lose — passed over."""
        rows = self.db.execute(
            "SELECT outcome, COUNT(*) c FROM offers WHERE last_at >= ? GROUP BY outcome",
            (since,)).fetchall()
        by = {r["outcome"]: int(r["c"]) for r in rows}
        total = sum(by.values())
        if not total:
            return ""
        return (f"Things you brought up unprompted: {total} raised, "
                f"{by.get('engaged', 0)} taken up by them, "
                f"{by.get('passed', 0)} passed over"
                + (f", {by.get('pending', 0)} not yet judged" if by.get("pending") else ""))


# ── the pool ─────────────────────────────────────────────────────────────────

def _rows(db, sql: str, params=()) -> list[dict]:
    try:
        return [dict(r) for r in db.execute(sql, params).fetchall()]
    except sqlite3.Error:
        return []


def candidates(memory, cfg: dict | None = None, now: float | None = None) -> list[dict]:
    """Everything she could bring up, newest first per source.

    Each item: {key, kind, text, topic, ts} — `text` is what she'd say it from,
    `topic` is what the segue test matches against."""
    cfg = cfg or {}
    now = now or time.time()
    kinds = set(cfg.get("kinds") or KINDS)
    pool = int(cfg.get("candidate_pool", 40))
    out: list[dict] = []

    if "news" in kinds and getattr(memory, "news", None) is not None:
        try:
            fresh = memory.news.search(since=now - float(cfg.get("news_max_age_s", 172800)),
                                       limit=pool)
        except Exception:
            fresh = []
        for r in fresh:
            title = (r.get("title") or "").strip()
            if not title:
                continue
            out.append({"key": f"news:{r.get('guid') or r.get('id')}", "kind": "news",
                        "text": title, "topic": f"{title} {r.get('summary') or ''}",
                        "ts": float(r.get("published_at") or r.get("fetched_at") or 0)})

    db = getattr(memory, "db", None)
    if db is not None and "finding" in kinds:
        # Research she FINISHED: the digest is her own summary of what she read.
        for r in _rows(db, "SELECT id, topic, title, digest FROM documents "
                           "WHERE kind='research' AND digest IS NOT NULL AND digest<>'' "
                           "AND fetched_at >= ? ORDER BY fetched_at DESC LIMIT ?",
                       (now - float(cfg.get("finding_max_age_s", 604800)), pool)):
            topic = (r.get("topic") or r.get("title") or "").strip()
            digest = (r.get("digest") or "").strip()
            if not topic or not digest:
                continue
            out.append({"key": f"finding:{r['id']}", "kind": "finding",
                        "text": f"what you found out about {topic}: {digest[:300]}",
                        "topic": f"{topic} {digest[:300]}", "ts": now})

    if db is not None and "question" in kinds:
        # …and research she HASN'T finished.  An open question is the most
        # conversational thing she holds: it's hers, it's honest, and it invites
        # the other person in instead of reciting at them.
        for r in _rows(db, "SELECT id, topic, query, reason FROM research_queue "
                           "WHERE status IN ('pending','failed') "
                           "ORDER BY id DESC LIMIT ?", (pool,)):
            topic = (r.get("topic") or "").strip()
            if not topic:
                continue
            out.append({"key": f"question:{r['id']}", "kind": "question",
                        "text": f"something you've been trying to find out: {topic}"
                                + (f" ({r['reason']})" if r.get("reason") else ""),
                        "topic": f"{topic} {r.get('query') or ''}", "ts": now})

    if "weather" in kinds and getattr(memory, "ambient", None) is not None:
        try:
            amb = [a for a in memory.ambient.active(now) if a.get("source") == "weather"]
        except Exception:
            amb = []
        for a in amb:
            payload = (a.get("payload") or "").strip()
            if payload and _NOTABLE_WEATHER.search(payload):
                out.append({"key": f"weather:{a.get('key') or payload[:40]}",
                            "kind": "weather", "text": payload, "topic": payload,
                            "ts": float(a.get("fetched_at") or now)})
    return out


# ── selection ────────────────────────────────────────────────────────────────

def shortlist(items: list[dict], user_text: str, log: OfferLog, cfg: dict | None = None,
              now: float | None = None) -> list[dict]:
    """The candidates that have a way in RIGHT NOW, best first.

    The test is topical overlap with what was just said.  Nothing overlapping
    means nothing is offered — there is no floor of "say something anyway",
    which is precisely the failure mode that makes assistants tiresome."""
    cfg = cfg or {}
    now = now or time.time()
    if not items or not (cfg.get("enabled", True)):
        return []
    if is_acknowledgement(user_text):
        return []                       # 'mm' gives nothing to segue FROM
    gap = float(cfg.get("min_gap_s", 900))
    if gap and now - log.last_at() < gap:
        return []                       # she just raised something; let it breathe
    cap = int(cfg.get("max_per_day", 6))
    if cap and log.count_since(now - 86400) >= cap:
        return []
    seen = log.spoken_keys()
    here = words(user_text)
    if not here:
        return []
    need = int(cfg.get("min_overlap", 2))
    scored = []
    for it in items:
        if it["key"] in seen:
            continue
        overlap = here & words(it.get("topic") or it.get("text") or "")
        if len(overlap) < need:
            continue
        scored.append((len(overlap), it.get("ts") or 0, it))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return [it for _, _, it in scored[:max(1, int(cfg.get("max_items", 1)))]]


def block(items: list[dict]) -> str:
    """The prompt block.  Written so that saying nothing stays a real option:
    the moment 'you have something to share' reads as an instruction, every
    conversation acquires an awkward tail."""
    if not items:
        return ""
    lines = [f"- ({it['kind']}) {it['text']}" for it in items]
    return ("\n\nSomething you've been holding that touches what they just said. Bring it "
            "up ONLY if there's a natural way in — as your own aside, in one or two "
            "sentences, the way someone mentions what they've been reading. If it "
            "doesn't genuinely connect, drop it and say nothing about it; there will be "
            "other moments, and a forced segue is worse than none. Never read it out as "
            "a headline, never list more than one, and don't announce that you have "
            "something to share:\n" + "\n".join(lines))


# ── outcomes ─────────────────────────────────────────────────────────────────

def was_spoken(item: dict, reply: str, min_share: float = 0.34) -> bool:
    """Did she actually work it into the reply?  A candidate she passed over
    must NOT be burnt — otherwise one crowded moment silently consumes it."""
    topic = words(item.get("text") or "")
    if not topic:
        return False
    hit = topic & words(reply)
    return len(hit) >= max(2, int(len(topic) * min_share))


def engaged_with(item: dict, user_text: str) -> bool:
    """Did they take it up?  A bare acknowledgement is a no — politeness is not
    interest, and treating it as interest is how an assistant learns to prattle."""
    if is_acknowledgement(user_text) or not (user_text or "").strip():
        return False
    here = words(user_text)
    if here & words(item.get("text") or ""):
        return True
    return "?" in user_text and len(here) >= 2     # they asked something back
