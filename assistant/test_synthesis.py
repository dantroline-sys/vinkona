#!/usr/bin/env python
"""
Tests for non-destructive personal-fact synthesis (memory.synthesize_profile /
_synthesize_theme) and the embed task-prefix migration.  numpy + aiohttp are
stubbed; a real temp SQLite backs the store.  The embedding-clustering maths in
synthesize_profile() needs real numpy, so (like the consolidate tests) it's left
for on-box integration — here we exercise the per-theme write path and guards
directly via _synthesize_theme with explicit ids.

    python test_synthesis.py
"""

import asyncio
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).parent

sys.modules.setdefault("numpy", types.ModuleType("numpy"))   # clustering maths not exercised here

# A test sets _RESP["text"] for the synthesis note the big LM "returns".
_RESP = {"text": "You're an outdoors person: you hike, climb, and love being in the hills."}
_EMB_INPUTS: list = []


class _FakeResp:
    def __init__(self, status, body): self.status, self._body = status, body
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def json(self): return self._body
    async def text(self): return json.dumps(self._body)


class _FakeSession:
    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    def post(self, url, json=None, timeout=None, **k):
        if url.endswith("/v1/chat/completions"):
            return _FakeResp(200, {"choices": [{"message": {"content": _RESP["text"]}}]})
        _EMB_INPUTS.append((json or {}).get("input"))
        return _FakeResp(200, {"data": [{"embedding": []}]})        # embed → None (stub numpy)


aiohttp_stub = types.ModuleType("aiohttp")
aiohttp_stub.ClientSession = _FakeSession
aiohttp_stub.ClientTimeout = lambda **k: None
sys.modules["aiohttp"] = aiohttp_stub

spec = importlib.util.spec_from_file_location("memory", HERE / "memory.py")
memory = importlib.util.module_from_spec(spec); spec.loader.exec_module(memory)

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}")


def _store(synthesis=None):
    tmp = tempfile.mkdtemp()
    cfg = {
        "memory": {"db_path": str(Path(tmp) / "m.db"), "recall_top_k": 5,
                   "recency_halflife_s": 1209600, "default_cooldown_s": 600, "min_score": 0.5,
                   "weights": {"priority": 0.5, "trigger": 2, "semantic": 1.5, "recency": 0.3,
                               "tag": 0.5, "cooldown_override_priority": 8},
                   "neighbours": 0, "neighbour_min_sim": 0.65, "garden": {},
                   "synthesis": synthesis or {}},
        "embed_lm": {"url": "http://x", "model": "m"},
    }
    return memory.MemoryStore(cfg)


def _insert(m, mid, payload, trigger="x", source="reflection", category="profile",
            priority=2, expiry=None):
    m.db.execute(
        "INSERT INTO memories(id,triggers,context_tags,payload,priority,recency,last_used,"
        "created_at,category,expiry,source,cooldown_until,embedding) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (mid, json.dumps([trigger]), "[]", payload, priority, 0.0, 0.0, 0.0,
         category, expiry, source, 0.0, None))
    m.db.commit()


def _synth_notes(m):
    return [dict(r) for r in m.db.execute(
        "SELECT * FROM memories WHERE source='synthesis'")]


# ── eligibility guard ──────────────────────────────────────────────────────────
def test_eligible():
    m = _store()
    elig = lambda e: m._synthesis_eligible(e, allow_crawl=False, now=1000.0)
    check("trusted personal profile is eligible",
          elig({"category": "profile", "source": "reflection"}))
    check("user-stated preference is eligible",
          elig({"category": "preference", "source": "user"}))
    check("world knowledge is NOT eligible",
          not elig({"category": "knowledge", "source": "research:wiki"}))
    check("a synthesis note never re-feeds synthesis",
          not elig({"category": "profile", "source": "synthesis"}))
    check("crawl source excluded when allow_crawl=False",
          not elig({"category": "profile", "source": "crawl:mail"}))
    check("crawl source included when allow_crawl=True",
          m._synthesis_eligible({"category": "profile", "source": "crawl:mail"},
                                allow_crawl=True, now=1000.0))
    check("expired fact excluded",
          not elig({"category": "fact", "source": "user", "expiry": 500.0}))


# ── the core promise: synthesis is NON-destructive ──────────────────────────────
async def test_non_destructive_write():
    m = _store({"enabled": True, "priority": 6})
    _insert(m, "a", "You enjoy hill-walking.", "hiking")
    _insert(m, "b", "You went climbing in the Lakes.", "climbing")
    _insert(m, "c", "You like being outdoors.", "outdoors")
    m.reload()
    res = await m._synthesize_theme(["a", "b", "c"], "http://big", "model", None,
                                    priority=6, cooldown_s=86400, now=1000.0)
    check("a new synthesis note is written", res.get("written") == 1)
    notes = _synth_notes(m)
    check("exactly one synthesis note exists", len(notes) == 1)
    check("note is category profile, source synthesis",
          notes and notes[0]["category"] == "profile" and notes[0]["source"] == "synthesis")
    check("note carries the integrated payload",
          notes and "outdoors person" in notes[0]["payload"])
    check("source facts are NOT deleted",
          all(m.db.execute("SELECT 1 FROM memories WHERE id=?", (i,)).fetchone()
              for i in ("a", "b", "c")))
    tags = json.loads(notes[0]["context_tags"]) if notes else []
    check("note links back to its member ids",
          {"mem:a", "mem:b", "mem:c"} <= set(tags) and "synth" in tags)


# ── refresh, not duplicate ──────────────────────────────────────────────────────
async def test_refresh_not_duplicate():
    m = _store({"enabled": True})
    _insert(m, "a", "You enjoy hill-walking.")
    _insert(m, "b", "You went climbing in the Lakes.")
    _insert(m, "c", "You like being outdoors.")
    m.reload()
    await m._synthesize_theme(["a", "b", "c"], "http://big", "model", None, 6, 86400, 1000.0)
    m.reload()
    first_id = _synth_notes(m)[0]["id"]
    # Within cooldown, unchanged membership → no-op.
    res2 = await m._synthesize_theme(["a", "b", "c"], "http://big", "model", None,
                                     6, 86400, 1100.0)
    check("fresh, unchanged theme is left alone", res2 == {})
    check("still exactly one synthesis note", len(_synth_notes(m)) == 1)
    # Membership changes → regenerate in place, reusing the same note id (no duplicate).
    _insert(m, "d", "You camp in the summer.")
    m.reload()
    res3 = await m._synthesize_theme(["a", "b", "c", "d"], "http://big", "model", None,
                                     6, 86400, 1200.0)
    check("changed-membership theme is updated, not added", res3.get("updated") == 1)
    notes = _synth_notes(m)
    check("no duplicate note created on refresh",
          len(notes) == 1 and notes[0]["id"] == first_id)


# ── trust-laundering guard ──────────────────────────────────────────────────────
async def test_taint_caps_priority():
    m = _store({"enabled": True, "priority": 6, "allow_crawl_sources": True})
    _insert(m, "a", "You enjoy hill-walking.", source="reflection")
    _insert(m, "b", "You went climbing in the Lakes.", source="crawl:mail")   # untrusted
    _insert(m, "c", "You like being outdoors.", source="user")
    m.reload()
    await m._synthesize_theme(["a", "b", "c"], "http://big", "model", None, 6, 86400, 1000.0)
    note = _synth_notes(m)[0]
    check("a crawl-tainted synthesis is capped low", note["priority"] <= 4)
    check("tainted synthesis is labelled", "crawl_tainted" in json.loads(note["context_tags"]))


# ── perspective guard: a swapped-voice synthesis is dropped ─────────────────────
async def test_perspective_guard():
    m = _store({"enabled": True})
    _insert(m, "a", "You enjoy hill-walking.")
    _insert(m, "b", "You went climbing in the Lakes.")
    _insert(m, "c", "You like being outdoors.")
    m.reload()
    _RESP["text"] = "I love hiking and being in the hills myself."     # first-person → wrong voice
    try:
        res = await m._synthesize_theme(["a", "b", "c"], "http://big", "model", None,
                                        6, 86400, 1000.0)
    finally:
        _RESP["text"] = "You're an outdoors person: you hike, climb, and love the hills."
    check("a first-person (swapped-voice) synthesis is refused", res == {})
    check("no note written when synthesis fails the perspective guard",
          len(_synth_notes(m)) == 0)


# ── synthesis never writes the People canon ─────────────────────────────────────
async def test_never_touches_canon():
    m = _store({"enabled": True})
    _insert(m, "a", "You enjoy hill-walking.")
    _insert(m, "b", "You went climbing in the Lakes.")
    _insert(m, "c", "You like being outdoors.")
    m.reload()
    await m._synthesize_theme(["a", "b", "c"], "http://big", "model", None, 6, 86400, 1000.0)
    # The note lives in `memories`, not the privileged people store.
    check("synthesis writes to memories, not people canon",
          _synth_notes(m) and m.db.execute(
              "SELECT COUNT(*) c FROM person_attributes WHERE provenance='synthesis'"
          ).fetchone()["c"] == 0)


# ── embed task-prefix + migration ───────────────────────────────────────────────
async def test_embed_prefix_and_migration():
    m = _store()
    m.embed_task_prefix = True
    m.embed_prefixes = {"query": "search_query: ", "document": "search_document: "}
    _EMB_INPUTS.clear()
    await m.embed("the user likes tea", task="document")
    check("document embed gets the search_document: prefix",
          _EMB_INPUTS and _EMB_INPUTS[-1] == "search_document: the user likes tea")
    await m.embed("tea?", task="query")
    check("query embed gets the search_query: prefix",
          _EMB_INPUTS[-1] == "search_query: tea?")
    # ensure_embed_format flips the stored format tag exactly once.
    _insert(m, "a", "You like tea.")
    m.reload()
    check("format starts raw", (m.get_state("embed_format") or "raw-v1") == "raw-v1")
    await m.ensure_embed_format()
    check("format becomes prefixed after migration",
          m.get_state("embed_format") == memory.EMBED_FORMAT)
    check("a second migration is a no-op", await m.ensure_embed_format() == 0)


async def main():
    test_eligible()
    await test_non_destructive_write()
    await test_refresh_not_duplicate()
    await test_taint_caps_priority()
    await test_perspective_guard()
    await test_never_touches_canon()
    await test_embed_prefix_and_migration()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    asyncio.run(main())
