#!/usr/bin/env python
"""Tests for spontaneity.py — the segue lane: what Vinkona is holding, whether
there's a way into it, and what happened when she raised it.  Real temp sqlite."""
import importlib.util
import sqlite3
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


sp = _load("spontaneity")

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}")


class _Ambient:
    def __init__(self, rows): self.rows = rows
    def active(self, now=None): return self.rows


class _News:
    def __init__(self, rows): self.rows = rows
    def search(self, since=None, limit=20, **kw):
        return [r for r in self.rows
                if (r.get("published_at") or 0) >= (since or 0)][:limit]


class _Memory:
    """Just enough MemoryStore for the candidate pass."""
    def __init__(self, db, news=None, ambient=None):
        self.db = db; self.news = news; self.ambient = ambient
        self.offers = sp.OfferLog(db)


def _db():
    c = sqlite3.connect(tempfile.mktemp(suffix=".db"))
    c.row_factory = sqlite3.Row
    c.executescript("""
      CREATE TABLE documents (id TEXT PRIMARY KEY, url TEXT, title TEXT, topic TEXT,
        fetched_at REAL, text TEXT, digest TEXT, kind TEXT);
      CREATE TABLE research_queue (id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT, topic TEXT, query TEXT, reason TEXT, status TEXT,
        attempts INTEGER, created_at REAL, updated_at REAL);
    """)
    return c


def test_words_and_acks():
    check("distinctive words drop stopwords and short tokens",
          sp.words("The weather is really about to turn cold")
          == {"weather", "turn", "cold"})
    for ack in ("mm", "okay", "yeah sure", "right, ok"):
        check(f"'{ack}' reads as an acknowledgement", sp.is_acknowledgement(ack))
    check("a real sentence is not an acknowledgement",
          not sp.is_acknowledgement("okay but what about the frost tonight"))


def test_candidates():
    now = time.time()
    db = _db()
    db.execute("INSERT INTO documents(id,topic,digest,kind,fetched_at) VALUES "
               "('d1','tomato blight','Copper sprays slow it but do not cure it.',"
               "'research',?)", (now - 3600,))
    db.execute("INSERT INTO documents(id,topic,digest,kind,fetched_at) VALUES "
               "('d2','old thing','stale digest','research',?)", (now - 40 * 86400,))
    db.execute("INSERT INTO research_queue(topic,query,reason,status,created_at) "
               "VALUES ('why sourdough starters stall','starter stall','you asked',"
               "'pending',?)", (now,))
    news = _News([{"guid": "g1", "title": "Frost warning for the south",
                   "summary": "Growers told to cover crops", "published_at": now - 3600},
                  {"guid": "g2", "title": "Ancient news", "summary": "",
                   "published_at": now - 30 * 86400}])
    amb = _Ambient([{"source": "weather", "key": "now", "payload": "18C, light cloud",
                     "fetched_at": now},
                    {"source": "weather", "key": "warn", "payload": "Storm warning tonight",
                     "fetched_at": now},
                    {"source": "calendar", "key": "c", "payload": "Dentist at 3",
                     "fetched_at": now}])
    mem = _Memory(db, news, amb)
    keys = {c["key"]: c for c in sp.candidates(mem, {}, now)}

    check("a fresh headline is a candidate", "news:g1" in keys)
    check("a stale headline is not", "news:g2" not in keys)
    check("a finished research finding is a candidate", "finding:d1" in keys)
    check("a week-old finding has aged out", "finding:d2" not in keys)
    check("an UNANSWERED research question is a candidate too (Dan's addition)",
          any(k.startswith("question:") for k in keys))
    check("the open question is phrased as hers, not as a fact",
          "trying to find out" in keys["question:1"]["text"])
    # ordinary weather is already in the ambient block every turn — repeating it
    # is not conversation; only the notable kind is worth raising
    check("notable weather is a candidate", "weather:warn" in keys)
    check("ordinary weather is not", "weather:now" not in keys)
    check("non-weather ambient rows are ignored",
          not any(k.startswith("weather:") and "Dentist" in keys[k]["text"] for k in keys))


def test_segue_gate():
    now = time.time()
    db = _db()
    log = sp.OfferLog(db)
    items = [{"key": "news:g1", "kind": "news", "text": "Frost warning for the south",
              "topic": "Frost warning for the south growers cover crops", "ts": now},
             {"key": "news:g2", "kind": "news", "text": "Ferry strike continues",
              "topic": "Ferry strike continues in the islands", "ts": now}]
    cfg = {"min_gap_s": 0}

    picked = sp.shortlist(items, "I need to cover the crops before the frost", log, cfg, now)
    check("a genuine topical overlap produces an offer",
          [p["key"] for p in picked] == ["news:g1"])
    check("an unrelated turn produces NOTHING (no 'say something anyway' floor)",
          sp.shortlist(items, "can you set a timer for ten minutes", log, cfg, now) == [])
    check("a single shared word is not a segue",
          sp.shortlist(items, "the south is nice", log, cfg, now) == [])
    check("an acknowledgement gives nothing to segue from",
          sp.shortlist(items, "mm, right", log, cfg, now) == [])
    check("only one thing is ever put in front of her", len(picked) == 1)
    check("disabled means silent",
          sp.shortlist(items, "frost on the crops", log, {"enabled": False}, now) == [])


def test_watermark_and_rate_limit():
    now = time.time()
    db = _db()
    log = sp.OfferLog(db)
    item = {"key": "news:g1", "kind": "news", "text": "Frost warning for the south",
            "topic": "frost warning south crops", "ts": now}
    items = [item]
    user = "there'll be a frost on the crops tonight"

    check("offered but NOT spoken leaves it available (a passed-over item isn't burnt)",
          sp.shortlist(items, user, log, {"min_gap_s": 0}, now)[0]["key"] == "news:g1")
    log.record(item, session_id="s1", now=now)
    check("once she has actually said it, it is never offered again",
          sp.shortlist(items, user, log, {"min_gap_s": 0}, now) == [])

    # rate limit: having just raised something, she gets nothing new for a while
    other = [{"key": "news:g9", "kind": "news", "text": "Frost damages vineyards",
              "topic": "frost vineyards crops", "ts": now}]
    check("she doesn't raise a second thing straight after the first",
          sp.shortlist(other, user, log, {"min_gap_s": 900}, now) == [])
    check("…and can again once it's had time to breathe",
          sp.shortlist(other, user, log, {"min_gap_s": 900}, now + 1000) != [])
    check("a daily cap is honoured",
          sp.shortlist(other, user, log, {"min_gap_s": 0, "max_per_day": 1}, now) == [])


def test_spoken_and_outcome():
    now = time.time()
    db = _db()
    log = sp.OfferLog(db)
    item = {"key": "news:g1", "kind": "news", "text": "Frost warning for the south tonight",
            "topic": "frost warning south", "ts": now}

    check("a reply that works it in counts as spoken",
          sp.was_spoken(item, "Worth covering them — there's a frost warning for the "
                              "south tonight."))
    check("a reply that ignored it does not",
          not sp.was_spoken(item, "Sure, I've set that timer for you."))

    log.record(item, session_id="s1", now=now)
    pend = log.pending("s1", now=now)
    check("the raised item is pending judgement", pend and pend["key"] == "news:g1")
    check("engagement is taking it up, not being polite about it",
          sp.engaged_with(item, "a frost? I'd better fleece the beds"))
    check("'mm' is not engagement — politeness is not interest",
          not sp.engaged_with(item, "mm"))
    check("a question back counts", sp.engaged_with(item, "how cold is it meant to get?"))

    log.resolve("news:g1", "engaged")
    check("a judged item is no longer pending", log.pending("s1", now=now) is None)
    s = log.summary(0)
    check("the summary reports it", "1 raised" in s and "1 taken up" in s)

    # the record must stay two-sided: passes are counted, not quietly dropped
    log.record({"key": "news:g2", "kind": "news", "text": "Ferry strike"}, "s1", now)
    log.resolve("news:g2", "passed")
    s2 = log.summary(0)
    check("passed-over offers are counted too", "1 passed over" in s2)
    check("nothing raised yet means no line at all", sp.OfferLog(_db()).summary(0) == "")


def test_bridge_wiring():
    """The three hooks actually fire on a turn: block into the prompt, spoken
    check on the reply, outcome judged on the NEXT user turn."""
    import asyncio
    import sys
    import types
    # llm_bridge imports aiohttp/numpy at module top; neither is touched on this
    # path, so stub them and run on a bare interpreter (test_reasoning_research
    # does the same).
    sys.modules.setdefault("numpy", types.ModuleType("numpy"))
    if "aiohttp" not in sys.modules:
        stub = types.ModuleType("aiohttp")
        stub.ClientSession = type("S", (), {})
        stub.ClientTimeout = lambda **k: None
        stub.ClientConnectorError = type("ClientConnectorError", (Exception,), {})
        sys.modules["aiohttp"] = stub
    bridge = _load("llm_bridge")
    seen = {"judged": [], "spoken": []}
    b = bridge.LLMBridge(
        server_state=types.SimpleNamespace(), fast_lm_url="http://f", big_lm_url=None,
        speak_sink=lambda *a, **k: None, inject_time=False, confirm_required=False,
        offer_hook=lambda ut: "\n\nOFFER-BLOCK for: " + ut,
        offer_spoken_hook=lambda reply: seen["spoken"].append(reply),
        offer_judge_hook=lambda ut: seen["judged"].append(ut))
    b._trace = lambda *a, **k: None
    captured = {}

    async def fake_run_turn(messages, tools):
        captured["system"] = messages[0]["content"]
        return "There's a frost warning tonight, as it happens."
    b._run_turn = fake_run_turn
    asyncio.run(b._handle_turn("I'm covering the crops"))

    check("the offer block reaches the fast LM's system prompt",
          "OFFER-BLOCK for: I'm covering the crops" in captured.get("system", ""))
    check("the reply is checked for what she actually said",
          seen["spoken"] == ["There's a frost warning tonight, as it happens."])
    check("the previous offer is judged by the user's next turn",
          seen["judged"] == ["I'm covering the crops"])


def main():
    test_words_and_acks()
    test_candidates()
    test_segue_gate()
    test_watermark_and_rate_limit()
    test_spoken_and_outcome()
    test_bridge_wiring()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
