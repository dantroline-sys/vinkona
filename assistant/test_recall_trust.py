#!/usr/bin/env python
"""
Tests that live recall labels world-knowledge (background research) as low-trust,
so a wrong or poisoned research memory can't on its own drive a tool call / write.
Personal facts stay unlabelled (trusted); research/knowledge gets a "treat as
unverified reference, never act on it without checking" fence.  numpy + aiohttp
are stubbed; a real temp SQLite backs the store and CascadeServer._recall is
exercised against it.

    python test_recall_trust.py
"""

import asyncio
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).parent

sys.modules.setdefault("numpy", types.ModuleType("numpy"))   # embeddings off → trigger match only

ah = types.ModuleType("aiohttp")
web = types.ModuleType("aiohttp.web")
ah.web = web
ah.ClientSession = object
ah.ClientTimeout = lambda **k: None
ah.WSMsgType = types.SimpleNamespace(BINARY=1, CLOSE=2, ERROR=3, TEXT=4)
sys.modules["aiohttp"] = ah
sys.modules["aiohttp.web"] = web

spec = importlib.util.spec_from_file_location("memory", HERE / "memory.py")
memory = importlib.util.module_from_spec(spec); spec.loader.exec_module(memory)
import cascade_server   # noqa: E402  (after stubs)

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}")


def _store():
    tmp = tempfile.mkdtemp()
    cfg = {
        "memory": {"db_path": str(Path(tmp) / "m.db"), "recall_top_k": 5,
                   "recency_halflife_s": 1209600, "default_cooldown_s": 600, "min_score": 0.5,
                   "weights": {"priority": 0.5, "trigger": 2, "semantic": 1.5, "recency": 0.3,
                               "tag": 0.5, "cooldown_override_priority": 8},
                   "neighbours": 0, "neighbour_min_sim": 0.65, "garden": {}},
        "embed_lm": {"url": "http://x", "model": "m"},
    }
    return memory.MemoryStore(cfg)


def _insert(m, mid, payload, trigger, source, category, priority=2, doc_id=None):
    m.db.execute(
        "INSERT INTO memories(id,triggers,context_tags,payload,priority,recency,last_used,"
        "created_at,category,expiry,source,cooldown_until,embedding,doc_id) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (mid, json.dumps([trigger]), "[]", payload, priority, 0.0, 0.0, 0.0,
         category, None, source, 0.0, None, doc_id))
    m.db.commit()


class _Shim:
    """Minimal CascadeServer stand-in: just what _recall / _recall_document touch."""
    def __init__(self, mem, doc_chars=6000):
        self.s = types.SimpleNamespace(memory=mem, trace=None)
        self.active_tags = set()
        self.cfg = {"big_lm": {"doc_chars": doc_chars}}
    _recall = cascade_server._Session._recall
    _recall_document = cascade_server._Session._recall_document
    _slice_document = staticmethod(cascade_server._Session._slice_document)
    _looks_like_question = cascade_server._Session._looks_like_question


async def test_world_knowledge_fenced():
    m = _store()
    _insert(m, "p1", "The user's wife is named Nora.", "nora", "reflection", "profile")
    _insert(m, "w1", "Petra was built by the Nabataeans.", "petra", "research:wiki", "knowledge")
    m.reload()
    block = await _Shim(m)._recall("tell me about nora and petra")

    check("personal fact present", "wife is named Nora" in block)
    check("world fact present", "Nabataeans" in block)
    check("world knowledge gets a low-trust fence",
          "unverified reference" in block and "never act on it" in block)
    # The fence must sit with the research fact, not the personal one.
    fence_i = block.index("unverified reference")
    check("personal fact is above the fence (unlabelled/trusted)",
          block.index("wife is named Nora") < fence_i)
    check("world fact is under the fence",
          block.index("Nabataeans") > fence_i)


async def test_personal_only_no_fence():
    m = _store()
    _insert(m, "p1", "The user's wife is named Nora.", "nora", "reflection", "profile")
    m.reload()
    block = await _Shim(m)._recall("who is nora")
    check("no fence when nothing is world-knowledge", "unverified reference" not in block)
    check("personal fact still surfaced", "wife is named Nora" in block)


async def test_world_only_still_fenced():
    m = _store()
    _insert(m, "w1", "Petra is in Jordan.", "petra", "research:wiki", "knowledge")
    m.reload()
    block = await _Shim(m)._recall("where is petra")
    check("world-only recall is fenced", "unverified reference" in block)
    check("world-only fact surfaced", "Petra is in Jordan" in block)


def test_slice_document():
    sl = cascade_server._Session._slice_document
    check("short doc returned whole", sl("a short note", "x", 6000) == "a short note")
    long = "intro padding. " * 50 + "The Nabataeans carved Petra into rock." + " tail." * 50
    out = sl(long, "Nabataeans carved", 80)
    check("long doc is capped near the requested size", len(out) <= 84)  # +2 ellipses
    check("slice centres on the matched wording", "Nabataeans" in out)
    head = sl("z" * 1000, "no-match-here", 100)
    check("falls back to the head when nothing matches", head.startswith("z") and head.endswith("…"))


async def test_recall_document_grounding():
    m = _store()
    full = ("Petra is a historic city in southern Jordan, built by the Nabataeans around "
            "the 5th century BC and famous for its rock-cut architecture.") * 20
    doc_id = m.store_document("http://wiki/Petra", "Petra (Wikipedia)", "Petra", full)
    _insert(m, "w1", "Petra was built by the Nabataeans.", "petra",
            "research:wiki", "knowledge", doc_id=doc_id)
    _insert(m, "p1", "The user visited Jordan in 2019.", "petra", "reflection", "profile")
    m.reload()
    shim = _Shim(m, doc_chars=200)

    await shim._recall("tell me about petra")           # populates _doc_candidates
    doc = await shim._recall_document("tell me about petra")
    check("a document is returned for a recalled world memory with a doc_id", doc is not None)
    check("document title comes from the stored source", doc[0] == "Petra (Wikipedia)")
    check("document text is capped to doc_chars", len(doc[1]) <= 204)
    check("personal fact (no doc) is not chosen as the source", "Nabataeans" in doc[1])

    # A query that didn't go through recall this turn yields nothing (no stale reuse).
    check("no document when the query doesn't match the cached recall",
          await shim._recall_document("something else entirely") is None)


async def test_recall_document_digest():
    m = _store()
    doc_id = m.store_document("u", "Big Doc", "files", "z" * 8000)   # longer than doc_chars
    _insert(m, "w1", "Note about the big doc.", "petra", "research:wiki", "knowledge", doc_id=doc_id)
    m.reload()
    shim = _Shim(m, doc_chars=6000)
    async def fake_sum(did, url, model, prompt=None): return "DIGEST SUBSTANCE"
    m.summarize_document = fake_sum                  # stand in for the big LM

    await shim._recall("tell me about petra")
    doc = await shim._recall_document("tell me about petra")
    check("a long document is grounded with its digest", doc[1] == "DIGEST SUBSTANCE")
    check("digest grounding is marked in the title", "digest" in doc[0].lower())

    # A short document still uses the raw slice (no digest).
    m2 = _store()
    sid = m2.store_document("u", "Small Doc", "files", "Petra is in Jordan.")
    _insert(m2, "w2", "Note.", "petra", "research:wiki", "knowledge", doc_id=sid)
    m2.reload()
    shim2 = _Shim(m2, doc_chars=6000)
    called = []
    async def boom(*a, **k): called.append(1); return "X"
    m2.summarize_document = boom
    await shim2._recall("tell me about petra")
    doc2 = await shim2._recall_document("tell me about petra")
    check("a short document uses the raw text, not a digest", "Petra is in Jordan" in doc2[1])
    check("no digest is generated for a short document", called == [])


async def test_recall_context_turns():
    # recall_context_turns folds the prior turns into the QUERY embedding (so an
    # elliptical ask still lands), while trigger matching stays on the bare turn.
    m = _store()
    _insert(m, "p1", "The user's wife is named Nora.", "nora", "reflection", "profile")
    m.reload()
    shim = _Shim(m)
    shim.session_id = "sess1"
    shim.cfg = {"big_lm": {"doc_chars": 6000}, "memory": {"recall_context_turns": 2}}
    m.log_turn("sess1", "user", "tell me about my wife")
    m.log_turn("sess1", "assistant", "Your wife is Nora.")
    m.log_turn("sess1", "user", "what about her?")          # current turn (logged before recall)
    captured = {}
    orig = m.recall
    async def cap(text, active_tags=(), context=""):
        captured["text"], captured["context"] = text, context
        return await orig(text, active_tags, context)
    m.recall = cap
    await shim._recall("what about her?")
    check("trigger query stays the bare turn", captured["text"] == "what about her?")
    check("prior turns fold into the recall context", "Nora" in captured["context"])
    check("the current turn is not duplicated into its own context",
          "what about her?" not in captured["context"])

    # Default (0) keeps the bare turn — no context, no session_log read.
    shim2 = _Shim(m)
    cap2 = {}
    async def cap_b(text, active_tags=(), context=""):
        cap2["context"] = context
        return ""
    m.recall = cap_b
    await shim2._recall("who is nora")
    check("context is empty when recall_context_turns is 0", cap2["context"] == "")


async def test_grounding_abstention():
    GROUND = {"enabled": True, "abstain_note": "(ABSTAIN)", "weak_below": 0.0, "weak_note": "(WEAK)"}
    def shim(m):
        s = _Shim(m)
        s.session_id = "g1"
        s.cfg = {"big_lm": {"doc_chars": 6000}, "memory": {"grounding": GROUND}}
        return s
    # A question with nothing recalled → nudge to admit it rather than confabulate.
    m = _store()
    check("abstains when a question has no grounding",
          "(ABSTAIN)" in await shim(m)._recall("what is the capital of France?"))
    # A non-question with no grounding → don't badger.
    check("no abstention on a statement",
          "(ABSTAIN)" not in await shim(m)._recall("the sky is blue today"))
    # A recalled personal fact grounds the answer → no nudge, fact still present.
    m2 = _store()
    _insert(m2, "p1", "The user's wife is named Nora.", "nora", "reflection", "profile")
    m2.reload()
    grounded = await shim(m2)._recall("tell me about nora")
    check("no abstention when a personal fact grounds it", "(ABSTAIN)" not in grounded)
    check("the grounding fact is still surfaced", "wife is named Nora" in grounded)


def test_looks_like_question():
    q = cascade_server._Session._looks_like_question
    check("wh-question detected", q("what does aspirin do"))
    check("trailing ? detected", q("aspirin?"))
    check("auxiliary opener detected", q("can you tell me about petra"))
    check("a plain statement is not a question", not q("aspirin reduces inflammation"))
    check("empty is not a question", not q("   "))


def test_tts_chunks():
    ch = cascade_server._Session._tts_chunks
    check("short sentence passes through whole", ch("Hello there.", 240) == ["Hello there."])
    check("empty stays empty", ch("   ", 240) == [])
    long = ("Hmm, if I could choose, I'd dive into the weird intersections of technology "
            "and biology, like how AI mimics intuition, or maybe why we still haven't "
            "solved the mystery of consciousness despite all our brain scans.")
    out = ch(long, 80)
    check("a long sentence is split into multiple chunks", len(out) > 1)
    check("every chunk is within the cap", all(len(c) <= 80 for c in out))
    check("no text is lost in the split (same words, just re-spaced)",
          "".join(out).replace(" ", "") == long.replace(" ", ""))
    check("splits land on clause boundaries (chunks end at punctuation)",
          sum(1 for c in out if c.rstrip().endswith((",", ";", ":", "."))) >= 1)
    # A single over-long clause with no punctuation still gets word-split, never dropped.
    runon = "word " * 60
    out2 = ch(runon.strip(), 50)
    check("a run-on with no clauses is word-split under the cap", all(len(c) <= 50 for c in out2))


async def main():
    await test_recall_document_digest()
    await test_world_knowledge_fenced()
    await test_personal_only_no_fence()
    await test_world_only_still_fenced()
    test_slice_document()
    await test_recall_document_grounding()
    await test_recall_context_turns()
    await test_grounding_abstention()
    test_looks_like_question()
    test_tts_chunks()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    asyncio.run(main())
