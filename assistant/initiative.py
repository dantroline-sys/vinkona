"""
VIN-INIT-01 deterministic core (IN1+IN2): the initiative queue, scorer,
governors, and the LM-free feeders.  See initiative_spec.md.

Everything here is deterministic and offline-testable.  The model's only job —
verbalising ONE selected item, or dropping it — lives at the injection point
(IN3, llm_bridge); nothing in this module talks to an LM.

The queue is a table in memory.db (per-profile, clearable — §9); the store
takes an open sqlite handle like NewsStore does.  Grounding is mandatory: an
item with nothing behind it is refused at add() — initiative must never be
confabulated (§3).  Selection is fail-closed: grief-adjacent never opens,
uncorroborated news never opens, respond_only never opens.
"""
from __future__ import annotations

import json
import re
import time
import uuid

CHANNELS = ("open_loop", "backlog", "news", "self_state", "reflection")
SENSITIVITIES = ("none", "personal", "grief_adjacent")

DEFAULT_WEIGHTS = {"timeliness": 1.0, "relational": 0.8, "novelty": 0.5,
                   "sensitivity": 1.5, "fatigue": 0.9}

# Per-channel default expiry (seconds); open_loop expiry is window_end + grace.
DEFAULT_EXPIRY_S = {"news": 172_800, "self_state": 259_200,
                    "backlog": 3_888_000, "reflection": 1_209_600}
OPEN_LOOP_GRACE_S = 604_800            # §4.1: window close + 7 days, strict

MAX_QUEUE_DEFAULT = 12
_FRESH_HALFLIFE_S = 432_000            # no window → mild 5-day freshness decay

# §0.3: explicit invitations bypass the frequency gate.  Deterministic phrases,
# matched against a lowercased opener; deliberately narrow — a false positive
# turns an ordinary greeting into an uninvited monologue.
_INVITES = (
    "what's new", "whats new", "anything new", "what is new",
    "what shall we talk about", "what should we talk about",
    "tell me something", "any news", "what's happening", "whats happening",
    "what have you been up to", "what are you thinking about",
)

_WORD = re.compile(r"[a-z0-9][a-z0-9'-]+")


def is_invitation(opener: str) -> bool:
    low = " ".join((opener or "").lower().split())
    return any(p in low for p in _INVITES)


def _terms(text: str) -> set:
    return set(_WORD.findall((text or "").lower()))


def _overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


class InitiativeQueue:
    def __init__(self, db, cfg: dict | None = None):
        self.db = db
        c = cfg or {}
        self.max_queue = int(c.get("max_queue", MAX_QUEUE_DEFAULT))
        self.weights = {**DEFAULT_WEIGHTS, **(c.get("weights") or {})}
        self.expiry_s = {**DEFAULT_EXPIRY_S, **(c.get("expiry_s") or {})}
        self._migrate()

    def _migrate(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS initiative_items (
              id TEXT PRIMARY KEY,
              channel TEXT NOT NULL,
              pointer TEXT NOT NULL,
              relational_context TEXT DEFAULT '',
              grounding TEXT NOT NULL,          -- json list, never empty
              created REAL NOT NULL,
              expires REAL,
              window_start REAL,
              window_end REAL,
              fit REAL DEFAULT 0.0,             -- §0.2: feeder-computed graph fit
              sensitivity TEXT DEFAULT 'none',
              corroborated INTEGER DEFAULT 0,   -- news: >=2 distinct sources
              respond_only INTEGER DEFAULT 0,   -- never initiates; recall only
              raised_count INTEGER DEFAULT 0,
              deflections INTEGER DEFAULT 0,
              last_outcome TEXT DEFAULT 'unraised',
              last_raised_at REAL,
              status TEXT DEFAULT 'queued')     -- queued | retired
        """)
        self.db.commit()

    # ── write side ───────────────────────────────────────────────────────────
    def add(self, channel: str, pointer: str, *, relational_context: str = "",
            grounding: list | None = None, fit: float = 0.0,
            sensitivity: str = "none", corroborated: bool = False,
            respond_only: bool = False, expires: float | None = None,
            window: tuple | None = None, now: float | None = None) -> dict:
        now = time.time() if now is None else float(now)
        pointer = " ".join(str(pointer or "").split())[:240]
        if channel not in CHANNELS:
            return {"ok": False, "error": f"unknown channel {channel!r}"}
        if not pointer:
            return {"ok": False, "error": "an item needs a pointer"}
        if not grounding:
            # §3: ungrounded initiative is inadmissible, full stop.
            return {"ok": False, "error": "ungrounded — initiative must never "
                                          "be confabulated"}
        if sensitivity not in SENSITIVITIES:
            return {"ok": False, "error": f"unknown sensitivity {sensitivity!r}"}
        if sensitivity == "grief_adjacent":
            respond_only = True                # §6.2: respond, never initiate
        w_start, w_end = (window or (None, None))
        if expires is None:
            if w_end is not None:
                expires = float(w_end) + OPEN_LOOP_GRACE_S
            else:
                expires = now + self.expiry_s.get(channel, 604_800)
        self.expire(now)
        # Dedupe: a live item with a near-identical pointer on the same channel.
        key = _terms(pointer)
        for it in self._live(now):
            if it["channel"] == channel and _overlap(key, _terms(it["pointer"])) >= 0.8:
                return {"ok": False, "error": "a similar item is already queued",
                        "duplicate": True, "id": it["id"]}
        item_id = f"init-{uuid.uuid4().hex[:12]}"
        self.db.execute(
            "INSERT INTO initiative_items(id, channel, pointer, relational_context,"
            " grounding, created, expires, window_start, window_end, fit,"
            " sensitivity, corroborated, respond_only)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (item_id, channel, pointer, str(relational_context or "")[:300],
             json.dumps(list(grounding)), now, expires, w_start, w_end,
             float(fit), sensitivity, int(bool(corroborated)),
             int(bool(respond_only))))
        self.db.commit()
        self._prune(now)
        row = self.get(item_id)
        return ({"ok": True, "item": row} if row else
                {"ok": False, "error": "pruned on insert (queue is full of "
                                       "stronger items)"})

    def _prune(self, now: float):
        live = self._live(now)
        openers = [i for i in live if not i["respond_only"]]
        excess = len(openers) - self.max_queue
        if excess <= 0:
            return
        openers.sort(key=lambda i: (self.salience(i, now), -i["created"]))
        for it in openers[:excess]:
            self.db.execute("DELETE FROM initiative_items WHERE id=?", (it["id"],))
        self.db.commit()

    def expire(self, now: float | None = None):
        now = time.time() if now is None else float(now)
        self.db.execute(
            "DELETE FROM initiative_items WHERE expires IS NOT NULL AND expires < ?",
            (now,))
        self.db.commit()

    def clear(self, channel: str | None = None):
        """Privacy tab: the whole queue, or one channel."""
        if channel:
            self.db.execute("DELETE FROM initiative_items WHERE channel=?", (channel,))
        else:
            self.db.execute("DELETE FROM initiative_items")
        self.db.commit()

    # ── read side ────────────────────────────────────────────────────────────
    def get(self, item_id: str) -> dict | None:
        r = self.db.execute("SELECT * FROM initiative_items WHERE id=?",
                            (item_id,)).fetchone()
        return self._row(r) if r else None

    def _row(self, r) -> dict:
        d = dict(r)
        d["grounding"] = json.loads(d.get("grounding") or "[]")
        d["respond_only"] = bool(d["respond_only"])
        d["corroborated"] = bool(d["corroborated"])
        return d

    def _live(self, now: float) -> list:
        rows = self.db.execute(
            "SELECT * FROM initiative_items WHERE status='queued' AND"
            " (expires IS NULL OR expires >= ?)", (now,)).fetchall()
        return [self._row(r) for r in rows]

    def items(self, now: float | None = None) -> list:
        """Everything live, salience-annotated — the panel's inspect view."""
        now = time.time() if now is None else float(now)
        out = self._live(now)
        for it in out:
            it["salience"] = round(self.salience(it, now), 3)
        out.sort(key=lambda i: -i["salience"])
        return out

    def provenance(self, item_id: str) -> list:
        """§9: 'why did you ask me that?' — the truthful answer."""
        it = self.get(item_id)
        return it["grounding"] if it else []

    # ── scoring (§5) ─────────────────────────────────────────────────────────
    def salience(self, it: dict, now: float, *, recent_topics: set | None = None,
                 last_channel: str | None = None,
                 channel_weights: dict | None = None) -> float:
        w = self.weights
        # timeliness
        ws, we = it.get("window_start"), it.get("window_end")
        if ws is not None and now < ws:
            timeliness = 0.15                  # not yet its moment
        elif we is not None and now > we:
            over = (now - we) / OPEN_LOOP_GRACE_S
            timeliness = max(0.0, 0.6 * (1.0 - over))
        elif ws is not None or we is not None:
            timeliness = 1.0                   # inside its window
        else:
            age = max(0.0, now - float(it.get("created") or now))
            timeliness = 0.5 ** (age / _FRESH_HALFLIFE_S)
        # novelty penalties
        novelty_pen = 0.0
        if last_channel and it["channel"] == last_channel:
            novelty_pen += 0.5
        if recent_topics:
            novelty_pen += _overlap(_terms(it["pointer"]), set(recent_topics))
        # sensitivity
        sens_pen = {"none": 0.0, "personal": 0.35,
                    "grief_adjacent": 10.0}[it.get("sensitivity") or "none"]
        # fatigue
        fatigue = 0.6 * int(it.get("raised_count") or 0) \
            + 0.8 * int(it.get("deflections") or 0)
        cw = float((channel_weights or {}).get(it["channel"], 1.0))
        return cw * (w["timeliness"] * timeliness
                     + w["relational"] * float(it.get("fit") or 0.0)) \
            - w["novelty"] * novelty_pen \
            - w["sensitivity"] * sens_pen \
            - w["fatigue"] * fatigue

    # ── the selector + governors (§6) ────────────────────────────────────────
    def pick(self, now: float | None = None, *, opener: str = "",
             invitation: bool | None = None, p: float = 0.6, rng=None,
             session_raised: bool = False, last_channel: str | None = None,
             recent_topics: set | None = None,
             channel_weights: dict | None = None) -> dict | None:
        """The deterministic selector.  Returns ONE item or None.  Governors:
        never twice per conversation unless the first engaged and the user asks
        again (caller passes session_raised + the new opener's invitation);
        uninvited openers fire with probability p; fail-closed exclusions."""
        now = time.time() if now is None else float(now)
        if invitation is None:
            invitation = is_invitation(opener)
        if session_raised and not invitation:
            return None
        if not invitation:
            import random
            roll = (rng() if rng is not None else random.random())
            if roll > p:
                return None
        self.expire(now)
        best, best_s = None, 0.0               # an item must EARN the floor
        for it in self._live(now):
            if it["respond_only"] or it["sensitivity"] == "grief_adjacent":
                continue                       # §6.2: respond, never initiate
            if it["channel"] == "news" and not it["corroborated"]:
                continue                       # §6.2: thin reports never speak
            ws = it.get("window_start")
            if ws is not None and now < ws:
                continue                       # not yet its natural moment
            s = self.salience(it, now, recent_topics=recent_topics,
                              last_channel=last_channel,
                              channel_weights=channel_weights)
            if s > best_s or (best is not None and s == best_s
                              and it["created"] < best["created"]):
                best, best_s = it, s
        return best

    # ── outcomes (§7) ────────────────────────────────────────────────────────
    def record_outcome(self, item_id: str, outcome: str):
        """raised: spoken, reception unclear.  engaged: taken up (item retires —
        its job is done).  deflected: brushed off (twice retires).  dropped:
        the model declined for fit — returns to the queue UNDISCOUNTED."""
        it = self.get(item_id)
        if it is None or outcome == "dropped":
            return
        if outcome == "raised":
            self.db.execute(
                "UPDATE initiative_items SET raised_count=raised_count+1,"
                " last_outcome='unraised', last_raised_at=? WHERE id=?",
                (time.time(), item_id))
        elif outcome == "engaged":
            self.db.execute(
                "UPDATE initiative_items SET raised_count=raised_count+1,"
                " last_outcome='engaged', last_raised_at=?, status='retired'"
                " WHERE id=?", (time.time(), item_id))
        elif outcome == "deflected":
            retire = int(it.get("deflections") or 0) + 1 >= 2
            self.db.execute(
                "UPDATE initiative_items SET raised_count=raised_count+1,"
                " deflections=deflections+1, last_outcome='deflected',"
                " last_raised_at=?" + (", status='retired'" if retire else "")
                + " WHERE id=?", (time.time(), item_id))
        self.db.commit()


# ── IN2: the LM-free feeders ──────────────────────────────────────────────────
# Self-state (§4.4): completed work from the worker's trace feed → honest
# "I did a thing" pointers.  Deterministic renderers per event kind; anything
# unmapped is silently ignored (no invented aliveness).
def _r_toolsmith(e):
    if e.get("action") == "deployed" and e.get("name"):
        return (f"I put together a new tool for myself overnight — '{e['name']}'",
                "she built it herself and wants to mention it")
    if e.get("action") == "gap_report" and e.get("title"):
        return (f"I tried to build something ('{e['title']}') and hit a wall — "
                "I wrote up what's missing",
                "an honest limitation she found in herself")
    return None


def _r_graph_run(e):
    if e.get("ok") and e.get("name") and not e.get("forced"):
        return (f"my '{e['name']}' tool ran on its own and worked",
                "a small win from a tool she made")
    return None


def _r_export(e):
    return ("I finished writing up my research notes for the knowledge host",
            "background work she completed") if e.get("full") else None


_SELF_STATE_RENDERERS = {"toolsmith": _r_toolsmith, "graph_run": _r_graph_run,
                         "export": _r_export}


def feed_self_state(queue: InitiativeQueue, events: list,
                    now: float | None = None) -> int:
    """Trace events → self_state items.  Returns how many were added."""
    now = time.time() if now is None else float(now)
    added = 0
    for e in events or []:
        fn = _SELF_STATE_RENDERERS.get(str(e.get("kind") or ""))
        if fn is None:
            continue
        rendered = fn(e)
        if rendered is None:
            continue
        pointer, why = rendered
        ref = f"trace:{e.get('kind')}:{e.get('name') or e.get('title') or ''}"
        if queue.add("self_state", pointer, relational_context=why,
                     grounding=[ref], fit=0.2, now=now).get("ok"):
            added += 1
    return added


def feed_backlog(queue: InitiativeQueue, questions: list,
                 graph_terms: set | None = None,
                 now: float | None = None) -> int:
    """Open research-plan questions (§4.2) → backlog items.  `questions` rows
    come from memory.next_plan_questions('research', …): id/question/topic.
    Fit = overlap between the question and the mind graph's terms (§0.2)."""
    now = time.time() if now is None else float(now)
    added = 0
    for q in questions or []:
        text = str(q.get("question") or "").strip()
        if not text:
            continue
        fit = _overlap(_terms(text + " " + str(q.get("topic") or "")),
                       set(graph_terms or ()))
        pointer = f"I never did settle something I was curious about: {text}"
        why = (f"it came out of her own reading on {q['topic']}"
               if q.get("topic") else "an open question from her own reading")
        if queue.add("backlog", pointer, relational_context=why,
                     grounding=[f"plan_question:{q.get('id')}"],
                     fit=min(1.0, fit), now=now).get("ok"):
            added += 1
    return added
