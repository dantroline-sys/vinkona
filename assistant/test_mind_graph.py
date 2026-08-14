#!/usr/bin/env python
"""VIN durable mind-graph (chat-derived personal knowledge graph).
Runs on a bare interpreter (sqlite + stub extractor, no model): grounding-required folds,
the identity firewall, cardinality supersession, corroboration, recall context, reversible
retraction, checkpointing, and deterministic replay.

    python assistant/test_mind_graph.py
"""
import asyncio
import importlib.util
import sqlite3
from pathlib import Path

HERE = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


mg = _load("mind_graph")

PASS = FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  ok  {name}")
    else: FAIL += 1; print(f"FAIL  {name}")


def fresh_db():
    db = sqlite3.connect(":memory:")
    db.executescript("""CREATE TABLE chat_logs (id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT, ts REAL, role TEXT, text TEXT);""")
    return db


def add_turn(db, role, text):
    db.execute("INSERT INTO chat_logs(session_id,ts,role,text) VALUES('s',0,?,?)", (role, text))
    db.commit()


def stub(data):
    """An extract_fn that ignores the prompt and returns canned extraction `data`."""
    return lambda _prompt: data


def distill(g, data, now=1000.0):
    return asyncio.run(g.distill(stub(data), now=now))


def test_anchor_and_schema():
    g = mg.MindGraph(fresh_db(), user_label="Dan")
    row = g.db.execute("SELECT type, locked FROM kg_nodes WHERE id=?", (mg.USER_ID,)).fetchone()
    check("the user anchor exists and is locked", row is not None and row[0] == "user" and row[1] == 1)
    check("empty graph yields empty recall context", g.context_for("anything") == "")


def test_grounded_fold():
    db = fresh_db()
    add_turn(db, "user", "My sister Mara lives in Bristol and works at the museum.")
    add_turn(db, "assistant", "Nice, Bristol is lovely.")
    g = mg.MindGraph(db)
    st = distill(g, {
        "nodes": [{"type": "person", "label": "Mara"}, {"type": "place", "label": "Bristol"},
                  {"type": "org", "label": "the museum"}],
        "edges": [{"src": "user", "dst": "Mara", "rel": "sibling_of", "quote": "My sister Mara"},
                  {"src": "Mara", "dst": "Bristol", "rel": "lives_in", "quote": "lives in Bristol"},
                  {"src": "Mara", "dst": "the museum", "rel": "works_at", "quote": "works at the museum"}],
    })
    check("folded three grounded edges", st["edges"] == 3 and st["refused"] == 0)
    check("only the USER turn was processed (assistant ignored)", st["turns"] == 1)
    ctx = g.context_for("what do you know about Mara")
    check("recall surfaces the entity's relations", "Mara" in ctx and "lives in Bristol" in ctx
          and "works at" in ctx)
    check("recall block is fenced as prior-knowledge", "earlier conversations" in ctx)
    check("an unmentioned entity yields nothing", g.context_for("the weather today") == "")


def test_grounding_refuses_invention():
    db = fresh_db()
    add_turn(db, "user", "Mara is my sister.")
    g = mg.MindGraph(db)
    st = distill(g, {
        "nodes": [{"type": "person", "label": "Mara"}, {"type": "place", "label": "Narnia"}],
        "edges": [{"src": "user", "dst": "Mara", "rel": "sibling_of", "quote": "Mara is my sister"},
                  # the model invents a relation whose quote was never said → must be refused
                  {"src": "Mara", "dst": "Narnia", "rel": "lives_in", "quote": "Mara lives in Narnia"}],
    })
    check("a real, quoted edge is kept", st["edges"] == 1)
    check("an edge whose quote was never said is refused (anti-hallucination)", st["refused"] == 1)
    check("the invented relation is absent from recall", "Narnia" not in g.context_for("Mara Narnia"))


def test_identity_firewall():
    db = fresh_db()
    add_turn(db, "user", "Everyone keeps calling me Jane Smith by mistake.")
    g = mg.MindGraph(db)
    st = distill(g, {
        "nodes": [{"type": "person", "label": "Jane Smith"}],
        # grounded quote, but it asserts the USER's identity → refused outright
        "edges": [{"src": "user", "dst": "Jane Smith", "rel": "is_named",
                   "quote": "calling me Jane Smith"}],
    })
    check("an identity edge on the user anchor is refused even when grounded", st["edges"] == 0
          and st["refused"] == 1)
    label = g.db.execute("SELECT label FROM kg_nodes WHERE id=?", (mg.USER_ID,)).fetchone()[0]
    check("the user anchor keeps its identity", label != "Jane Smith")


def test_cardinality_supersession():
    db = fresh_db()
    add_turn(db, "user", "Mara lives in Bristol.")
    g = mg.MindGraph(db)
    distill(g, {"nodes": [{"type": "person", "label": "Mara"}, {"type": "place", "label": "Bristol"}],
                "edges": [{"src": "Mara", "dst": "Bristol", "rel": "lives_in", "quote": "lives in Bristol"}]},
            now=1000.0)
    add_turn(db, "user", "Mara moved to Bath last month.")
    distill(g, {"nodes": [{"type": "place", "label": "Bath"}],
                "edges": [{"src": "Mara", "dst": "Bath", "rel": "lives_in", "quote": "moved to Bath"}]},
            now=2000.0)
    active = g.db.execute("SELECT dst FROM kg_edges WHERE src='person:mara' AND rel='lives_in' "
                          "AND status='active' AND valid_to IS NULL").fetchall()
    check("only the current residence is active", [r[0] for r in active] == ["place:bath"])
    superseded = g.db.execute("SELECT dst FROM kg_edges WHERE rel='lives_in' AND status='superseded'").fetchall()
    check("the old residence is superseded, not deleted (history kept)",
          [r[0] for r in superseded] == ["place:bristol"])
    check("recall shows only the current home", "Bath" in g.context_for("where does Mara live")
          and "Bristol" not in g.context_for("where does Mara live"))


def test_corroboration_and_checkpoint():
    db = fresh_db()
    add_turn(db, "user", "My friend Sam is a chef.")
    g = mg.MindGraph(db)
    distill(g, {"nodes": [{"type": "person", "label": "Sam"}],
                "edges": [{"src": "user", "dst": "Sam", "rel": "friend_of", "quote": "My friend Sam"}]})
    # nothing new → a second distill is a no-op that advances nothing
    st2 = distill(g, {"nodes": [], "edges": []})
    check("distill is a no-op when there are no new user turns", st2["turns"] == 0)
    # a new turn re-asserting the same fact corroborates (mentions bump), not a duplicate edge
    add_turn(db, "user", "Sam my friend cooked dinner again.")
    distill(g, {"edges": [{"src": "user", "dst": "Sam", "rel": "friend_of", "quote": "Sam my friend"}]})
    m = g.db.execute("SELECT mentions FROM kg_edges WHERE src=? AND rel='friend_of'", (mg.USER_ID,)).fetchone()[0]
    n = g.db.execute("SELECT COUNT(*) FROM kg_edges WHERE rel='friend_of'").fetchone()[0]
    check("re-asserting a fact corroborates it (mentions ≥ 2)", m >= 2)
    check("corroboration does not create a duplicate edge", n == 1)


def test_reversible_retract():
    db = fresh_db()
    add_turn(db, "user", "Mara is my sister.")
    g = mg.MindGraph(db)
    distill(g, {"nodes": [{"type": "person", "label": "Mara"}],
                "edges": [{"src": "user", "dst": "Mara", "rel": "sibling_of", "quote": "Mara is my sister"}]})
    eid = mg._edge_key(mg.USER_ID, "sibling_of", "person:mara")
    check("retract flips the edge off", g.retract_edge(eid) is True)
    check("a retracted edge leaves recall", "sister" not in g.context_for("tell me about Mara").lower())
    check("the anchor can never be retracted", g.retract_node(mg.USER_ID) is False)


def test_deterministic_fold():
    data = {"nodes": [{"type": "person", "label": "Mara"}, {"type": "place", "label": "Bristol"}],
            "edges": [{"src": "user", "dst": "Mara", "rel": "sibling_of", "quote": "sister Mara"},
                      {"src": "Mara", "dst": "Bristol", "rel": "lives_in", "quote": "in Bristol"}]}
    def run():
        db = fresh_db()
        add_turn(db, "user", "My sister Mara is in Bristol.")
        g = mg.MindGraph(db)
        distill(g, data, now=1234.0)
        return sorted(g.db.execute("SELECT id,src,dst,rel FROM kg_edges").fetchall()), \
               sorted(g.db.execute("SELECT id,type,label FROM kg_nodes").fetchall())
    check("the fold is deterministic (same input → same graph)", run() == run())


def test_snapshot():
    db = fresh_db()
    add_turn(db, "user", "My sister Mara lives in Bristol.")
    g = mg.MindGraph(db)
    distill(g, {"nodes": [{"type": "person", "label": "Mara"}, {"type": "place", "label": "Bristol"}],
                "edges": [{"src": "user", "dst": "Mara", "rel": "sibling_of", "quote": "sister Mara"},
                          {"src": "Mara", "dst": "Bristol", "rel": "lives_in", "quote": "lives in Bristol"}]})
    snap = g.snapshot()
    import json as _j
    check("snapshot is JSON-serialisable", isinstance(_j.dumps(snap), str))
    check("snapshot carries nodes, edges, counts", snap["nodes"] and snap["edges"] and snap["counts"]["edges"] >= 2)
    check("the anchor is flagged locked in the snapshot",
          any(n["locked"] for n in snap["nodes"] if n["id"] == mg.USER_ID))


def _per_turn_stub():
    """extract_fn that grounds one edge in every user turn present in the prompt (quoting it
    verbatim), so a catch_up drain folds something from each batch regardless of size."""
    import re as _re
    def fn(prompt):
        nodes, edges = [], []
        for m in _re.finditer(r"^\[(\d+)\] (.+)$", prompt, _re.M):
            label = f"item{m.group(1)}"
            nodes.append({"type": "thing", "label": label})
            edges.append({"src": "user", "dst": label, "rel": "mentioned",
                          "quote": m.group(2).strip()[:20]})
        return {"nodes": nodes, "edges": edges}
    return fn


def test_no_duplicate_distilling():
    db = fresh_db()
    add_turn(db, "user", "My sister Mara lives in Bristol.")
    g = mg.MindGraph(db)
    data = {"nodes": [{"type": "person", "label": "Mara"}, {"type": "place", "label": "Bristol"}],
            "edges": [{"src": "user", "dst": "Mara", "rel": "sibling_of", "quote": "sister Mara"},
                      {"src": "Mara", "dst": "Bristol", "rel": "lives_in", "quote": "lives in Bristol"}]}
    asyncio.run(g.catch_up(stub(data)))
    e1 = g.db.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
    # re-running over already-distilled turns (no new user turns) must process nothing…
    st = asyncio.run(g.catch_up(stub(data)))
    e2 = g.db.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
    check("a re-run over already-distilled turns processes no turns", st["turns"] == 0)
    check("…and folds no batches", st["batches"] == 0)
    check("…and creates no duplicate edges", e1 == 2 and e2 == 2)
    check("stats exposes a zero backlog once caught up", g.stats()["backlog"] == 0)


def test_catch_up_backfills_old_chats():
    db = fresh_db()
    for i in range(5):                                   # a backlog of 5 old user turns…
        add_turn(db, "user", f"Historic fact number {i} that was said earlier.")
        add_turn(db, "assistant", "noted")               # …interleaved with assistant turns
    g = mg.MindGraph(db, {"distill_batch_turns": 1, "distill_max_batches": 3})
    check("backlog counts unprocessed USER turns only (assistant ignored)", g.backlog() == 5)
    st = asyncio.run(g.catch_up(_per_turn_stub()))
    check("one pass drains up to distill_max_batches batches", st["batches"] == 3)
    check("three old user turns were distilled this pass", st["turns"] == 3)
    check("the backlog shrank but is not yet empty", st["backlog"] == 2)
    st2 = asyncio.run(g.catch_up(_per_turn_stub()))
    check("the next pass drains the remainder", st2["turns"] == 2 and st2["backlog"] == 0)
    st3 = asyncio.run(g.catch_up(_per_turn_stub()))
    check("once caught up a pass is a no-op", st3["turns"] == 0 and st3["batches"] == 0)
    # a fresh turn arriving later is picked up next pass — not blocked, not re-doing the old
    add_turn(db, "user", "A brand new fact stated just now for the record.")
    st4 = asyncio.run(g.catch_up(_per_turn_stub()))
    check("a new turn is distilled without re-processing the old", st4["turns"] == 1)


def test_prompt_carries_a_grounded_worked_example():
    import json as _j
    g = mg.MindGraph(fresh_db())
    p = g.build_prompt([{"id": 7, "text": "hello"}])
    check("prompt embeds a worked example output (shape shown, not just described)",
          mg.MindGraph._EXAMPLE_OUT in p)
    check("prompt forbids prose/markdown so weak models return only JSON", "ONLY the JSON" in p)
    ex = _j.loads(mg.MindGraph._EXAMPLE_OUT)
    check("the example is valid JSON in the required schema",
          isinstance(ex.get("nodes"), list) and isinstance(ex.get("edges"), list)
          and all(k in e for e in ex["edges"] for k in ("src", "dst", "rel", "quote", "fact")))
    # the example must itself be GROUNDED (each quote occurs in the example input) — we teach
    # exactly the behaviour _fold enforces, so a model copying it produces edges that survive.
    low = mg.MindGraph._EXAMPLE_IN.lower()
    check("every example edge quote actually occurs in the example input (grounded)",
          all(e["quote"].lower() in low for e in ex["edges"]))
    check("prompt teaches the snippet→clean-fact paraphrase (traffic-lights example)",
          "The user likes traffic lights" in p and "traffic lights are awesome" in p)
    check("every example edge carries a clean paraphrased fact",
          all(e.get("fact", "").strip() for e in ex["edges"]))


def test_paraphrased_fact_is_surfaced_not_the_snippet():
    db = fresh_db()
    add_turn(db, "user", "honestly traffic lights are awesome, and they just make me happy")
    g = mg.MindGraph(db)
    distill(g, {"nodes": [{"type": "thing", "label": "traffic lights"}],
                "edges": [{"src": "user", "dst": "traffic lights", "rel": "likes",
                           "quote": "traffic lights are awesome",
                           "fact": "The user likes traffic lights"}]})
    ctx = g.context_for("what about traffic lights")
    check("recall surfaces the clean paraphrased fact", "The user likes traffic lights" in ctx)
    check("recall does NOT surface the raw snippet", "and they" not in ctx and "awesome" not in ctx)
    snap = g.snapshot()
    e = next(e for e in snap["edges"] if e["dst"] == "thing:traffic lights")
    check("the edge keeps BOTH the clean fact and the grounding quote (evidence)",
          e["fact"] == "The user likes traffic lights" and "awesome" in e["quote"])


def test_fact_absent_falls_back_to_triple():
    db = fresh_db()
    add_turn(db, "user", "My sister Mara lives in Bristol.")
    g = mg.MindGraph(db)
    distill(g, {"nodes": [{"type": "person", "label": "Mara"}, {"type": "place", "label": "Bristol"}],
                "edges": [{"src": "Mara", "dst": "Bristol", "rel": "lives_in",
                           "quote": "lives in Bristol"}]})   # no 'fact' → old/weak extraction
    ctx = g.context_for("tell me about Mara")
    check("an edge with no fact still reads as a plain triple", "Mara lives in Bristol" in ctx)


def test_corroboration_updates_the_fact():
    db = fresh_db()
    add_turn(db, "user", "I love traffic lights.")
    g = mg.MindGraph(db)
    distill(g, {"nodes": [{"type": "thing", "label": "traffic lights"}],
                "edges": [{"src": "user", "dst": "traffic lights", "rel": "likes",
                           "quote": "love traffic lights", "fact": "The user likes traffic lights"}]})
    add_turn(db, "user", "traffic lights are the best thing ever, I adore them")
    distill(g, {"edges": [{"src": "user", "dst": "traffic lights", "rel": "likes",
                           "quote": "traffic lights are the best",
                           "fact": "The user adores traffic lights"}]})
    e = next(e for e in g.snapshot()["edges"] if e["dst"] == "thing:traffic lights")
    check("corroboration keeps the latest clean fact", e["fact"] == "The user adores traffic lights")
    check("corroboration bumps mentions, not a duplicate edge", e["mentions"] >= 2)


def test_failed_extraction_does_not_advance():
    db = fresh_db()
    add_turn(db, "user", "My sister Mara lives in Bristol.")
    g = mg.MindGraph(db)
    # A FAILED extraction call (None: LM down / non-JSON) must NOT advance the checkpoint —
    # else the turn is silently 'processed' to nothing and can never be revisited.
    st = distill(g, None)
    check("a failed extraction is flagged, not folded", st.get("failed") is True and st["turns"] == 1)
    check("a failed extraction leaves the whole backlog intact", g.backlog() == 1)
    check("nothing was folded on failure (only the locked anchor exists, no edges)",
          g.stats()["nodes"] == 1 and g.stats()["edges"] == 0)
    # once the LM is back, the SAME turn distils normally (it was never checkpointed away)
    distill(g, {"nodes": [{"type": "person", "label": "Mara"}],
                "edges": [{"src": "user", "dst": "Mara", "rel": "sibling_of", "quote": "My sister Mara"}]})
    check("the retried turn folds and now advances", g.backlog() == 0 and g.stats()["edges"] == 1)
    # an EMPTY-but-successful extraction ({}) DOES advance (the LM genuinely found nothing)
    add_turn(db, "user", "Just chatting, nothing to note.")
    st2 = distill(g, {})
    check("an empty successful extraction advances (no infinite retry)",
          st2.get("failed") is None and g.backlog() == 0)


def test_catch_up_stops_on_failure():
    db = fresh_db()
    for i in range(4):
        add_turn(db, "user", f"Message number {i} to distil.")
    g = mg.MindGraph(db, {"distill_batch_turns": 1, "distill_max_batches": 6})
    st = asyncio.run(g.catch_up(stub(None)))            # every call fails
    check("catch_up stops on failure instead of looping the cap", st.get("failed") is True
          and st["batches"] == 0)
    check("catch_up preserves the full backlog on failure", st["backlog"] == 4)


def test_catch_up_all_drains_past_the_one_pass_ceiling():
    """The 'Redistil everything' bug: a forced rebuild ran ONE capped catch_up
    (batch_turns×max_batches turns — 1000 on the live box) and stopped.  catch_up_all
    must drain the WHOLE backlog, stand down cooperatively, and resume from the
    checkpoint without re-folding."""
    db = fresh_db()
    for i in range(10):
        add_turn(db, "user", f"Backlog message {i} for the rebuild.")
    # one pass folds at most 2×2 = 4 turns — the old ceiling
    g = mg.MindGraph(db, {"distill_batch_turns": 2, "distill_max_batches": 2})
    st = asyncio.run(g.catch_up_all(stub({})))
    check("catch_up_all drains PAST the per-pass ceiling (all 10, not 4)",
          st["turns"] == 10 and st["backlog"] == 0 and st.get("done") is True)
    # cooperative stand-down: yield after the first pass, resume later
    db2 = fresh_db()
    for i in range(10):
        add_turn(db2, "user", f"Second backlog message {i}.")
    g2 = mg.MindGraph(db2, {"distill_batch_turns": 2, "distill_max_batches": 2})
    calls = {"n": 0}
    def yield_after_first():
        calls["n"] += 1
        return calls["n"] >= 1                          # user comes back immediately
    st1 = asyncio.run(g2.catch_up_all(stub({}), should_yield=yield_after_first))
    check("catch_up_all stands down for the user (done=False, backlog preserved)",
          st1.get("done") is False and st1["turns"] == 4 and st1["backlog"] == 6)
    st2 = asyncio.run(g2.catch_up_all(stub({})))
    check("a resumed drain finishes from the checkpoint, re-folding nothing",
          st2.get("done") is True and st2["turns"] == 6 and st2["backlog"] == 0)
    # failure: stop, flag, preserve — never spin
    db3 = fresh_db()
    for i in range(3):
        add_turn(db3, "user", f"Third backlog message {i}.")
    g3 = mg.MindGraph(db3, {"distill_batch_turns": 1, "distill_max_batches": 1})
    st3 = asyncio.run(g3.catch_up_all(stub(None)))
    check("catch_up_all stops on failure with the backlog intact",
          st3.get("failed") is True and st3.get("done") is False and st3["backlog"] == 3)


def test_poison_turn_cannot_wedge_the_backlog():
    """The real 'stuck at 1000' bug: checkpoint-on-failure is right for a TRANSIENT outage
    but deadlocks on a DETERMINISTIC one — a turn whose text blows the context window fails
    identically every pass, so the same batch is retried forever and the backlog freezes at
    exactly the same number.  A failed batch must bisect, and a lone unextractable turn must
    be quarantined — but only when the model is provably answering."""
    db = fresh_db()
    for i in range(3):
        add_turn(db, "user", f"Ordinary message {i}.")
    add_turn(db, "user", "POISON " + ("x" * 500))          # the one that blows the prompt
    for i in range(3):
        add_turn(db, "user", f"Later message {i}.")
    g = mg.MindGraph(db, {"distill_batch_turns": 8, "distill_max_batches": 4})

    def extractor(healthy=True):
        """Fails on any prompt containing the poison turn; otherwise succeeds."""
        def fn(prompt):
            if not healthy:
                return None                                # the whole model is down
            return None if "POISON" in prompt else {}
        return fn

    # a batch containing the poison turn bisects, folds the good turns, quarantines the one
    st = asyncio.run(g.catch_up(extractor()))
    check("the backlog drains despite an unextractable turn", g.backlog() == 0)
    check("exactly one turn was quarantined", st.get("skipped") == 1)
    check("the good turns around it were still folded", st["turns"] >= 6)
    sk = g.skipped_turns()
    check("the quarantined turn is recorded for audit (id + size)",
          len(sk) == 1 and sk[0]["chars"] > 500)
    check("stats surfaces the skip count", g.stats()["skipped"] == 1)

    # a genuine outage must NOT skip anything — the backlog is preserved for a retry
    db2 = fresh_db()
    for i in range(4):
        add_turn(db2, "user", f"Message {i} during an outage.")
    g2 = mg.MindGraph(db2, {"distill_batch_turns": 4, "distill_max_batches": 2})
    st2 = asyncio.run(g2.catch_up(extractor(healthy=False)))
    check("a full outage preserves the whole backlog (nothing skipped)",
          st2.get("failed") is True and g2.backlog() == 4 and st2.get("skipped", 0) == 0)
    check("nothing was quarantined during the outage", g2.skipped_turns() == [])
    # …and once it recovers, those same turns fold normally
    st3 = asyncio.run(g2.catch_up(extractor()))
    check("after recovery the preserved backlog folds normally",
          g2.backlog() == 0 and st3["turns"] == 4)

    # an over-long turn is CLIPPED in the prompt, so most 'poison' never happens at all
    g3 = mg.MindGraph(fresh_db(), {"max_turn_chars": 50})
    long_prompt = g3.build_prompt([{"id": 1, "text": "y" * 5000}])
    check("a huge turn is clipped in the extraction prompt (body bounded, not the template)",
          ("y" * 51) not in long_prompt and ("y" * 50) in long_prompt)
    check("clipping is visible, not silent", "truncated" in long_prompt)
    g4 = mg.MindGraph(fresh_db(), {"max_turn_chars": 0})
    check("clipping can be disabled (0 = no cap)",
          len(g4.build_prompt([{"id": 1, "text": "y" * 5000}])) > 5000)


def test_value_coming_back_resurrects_not_crashes():
    """The SECOND stuck-at-N wedge (stock-take 2026-08-13): a card-one value that comes
    BACK (Bristol → London → Bristol) hit an INSERT on its old primary key — the row
    still existed as status='superseded' — and the IntegrityError escaped the fold,
    freezing the checkpoint on that batch forever.  Fresh grounded evidence must
    resurrect the old row, exactly as _resolve_node already does for nodes."""
    db = fresh_db()
    add_turn(db, "user", "I live in Bristol.")
    g = mg.MindGraph(db)
    distill(g, {"nodes": [{"type": "place", "label": "Bristol"}],
                "edges": [{"src": "user", "dst": "Bristol", "rel": "lives_in",
                           "quote": "live in Bristol"}]}, now=1000.0)
    add_turn(db, "user", "I moved to London.")
    distill(g, {"nodes": [{"type": "place", "label": "London"}],
                "edges": [{"src": "user", "dst": "London", "rel": "lives_in",
                           "quote": "moved to London"}]}, now=2000.0)
    add_turn(db, "user", "I moved back to Bristol.")
    st = distill(g, {"nodes": [{"type": "place", "label": "Bristol"}],
                     "edges": [{"src": "user", "dst": "Bristol", "rel": "lives_in",
                                "quote": "moved back to Bristol"}]}, now=3000.0)
    check("the come-back batch folds instead of crashing (nothing skipped)",
          st.get("skipped") == 0 and st.get("edges", 0) >= 1 and not st.get("failed"))
    active = g.db.execute(
        "SELECT dst, valid_from FROM kg_edges WHERE src=? AND rel='lives_in' "
        "AND status='active' AND valid_to IS NULL", (mg.USER_ID,)).fetchall()
    check("the returned value is the ONE active edge, with a fresh validity interval",
          [r[0] for r in active] == ["place:bristol"] and active[0][1] == 3000.0)
    check("the interim value is superseded, not deleted (history kept)",
          g.db.execute("SELECT COUNT(*) FROM kg_edges WHERE dst='place:london' "
                       "AND status='superseded'").fetchone()[0] == 1)
    check("the checkpoint advanced past the whole transcript", g._last_id() == 3)
    m = g.db.execute("SELECT mentions FROM kg_edges WHERE dst='place:bristol' "
                     "AND rel='lives_in'").fetchone()[0]
    check("resurrection corroborates the original row (same triple, not a twin)", m == 2)


def test_retracted_edge_resurrects_on_fresh_evidence():
    """Mirrors node semantics: a retracted node already comes back on re-mention
    (_resolve_node), so a retracted edge re-asserted with fresh grounded evidence
    comes back too — and must never crash the fold."""
    db = fresh_db()
    add_turn(db, "user", "Sam plays the violin.")
    g = mg.MindGraph(db)
    distill(g, {"nodes": [{"type": "person", "label": "Sam"}],
                "edges": [{"src": "Sam", "dst": "violin", "rel": "plays",
                           "quote": "plays the violin"}]}, now=1000.0)
    eid = g.db.execute("SELECT id FROM kg_edges WHERE rel='plays'").fetchone()[0]
    check("retract flips the edge off", g.retract_edge(eid)
          and g.db.execute("SELECT status FROM kg_edges WHERE id=?", (eid,)).fetchone()[0]
          == "retracted")
    add_turn(db, "user", "Sam plays the violin every Sunday.")
    st = distill(g, {"nodes": [{"type": "person", "label": "Sam"}],
                     "edges": [{"src": "Sam", "dst": "violin", "rel": "plays",
                                "quote": "plays the violin every Sunday"}]}, now=2000.0)
    check("re-asserting a retracted edge folds cleanly", not st.get("failed")
          and st.get("skipped") == 0)
    check("…and the edge is active again",
          g.db.execute("SELECT status FROM kg_edges WHERE id=?", (eid,)).fetchone()[0]
          == "active")


def test_fold_crash_cannot_wedge_the_backlog():
    """Belt-and-braces for the whole defect class: ANY exception escaping _fold used to
    freeze the checkpoint (only extract_fn was wrapped).  A fold crash now degrades like
    a failed extraction — bisect, then quarantine — so the backlog always advances."""
    db = fresh_db()
    add_turn(db, "user", "turn one")
    add_turn(db, "user", "turn two")
    g = mg.MindGraph(db)

    def boom(data, turns, now):
        raise RuntimeError("deliberate fold bug")
    g._fold = boom
    st = distill(g, {"nodes": [], "edges": []})
    check("a fold crash quarantines the turns instead of raising",
          st.get("skipped") == 2 and not st.get("failed"))
    check("the checkpoint stepped past both turns", g._last_id() == 2)
    check("the quarantine is auditable in stats", g.stats()["skipped"] == 2)


def test_non_latin_labels():
    """The old _NORM_STRIP deleted every non-ASCII char: "北京" normalised to "" (all
    such entities of one type collapsed into a single node, their edges refused) and
    "Reykjavík" mangled to "reykjav k"."""
    check("non-Latin labels keep their letters", mg._norm("北京") == "北京")
    check("accented labels keep their accents", mg._norm("Reykjavík") == "reykjavík")
    check("punctuation still strips", mg._norm("  Mara,  O'Brien! ") == "mara o brien")
    db = fresh_db()
    add_turn(db, "user", "My friend Wei lives in 北京 now.")
    g = mg.MindGraph(db)
    st = distill(g, {"nodes": [{"type": "person", "label": "Wei"},
                               {"type": "place", "label": "北京"}],
                     "edges": [{"src": "Wei", "dst": "北京", "rel": "lives_in",
                                "quote": "lives in 北京"}]})
    check("a non-Latin entity gets its own node and a grounded edge",
          st.get("edges", 0) == 1
          and g.db.execute("SELECT COUNT(*) FROM kg_nodes WHERE id='place:北京'")
                 .fetchone()[0] == 1)


def test_norm_migration_renames_mangled_keys():
    """A graph built under the ASCII-only _norm carries mangled keys ("person:reykjav k");
    opening it under the fixed _norm must rename node + edge keys in place, once."""
    db = fresh_db()
    g = mg.MindGraph(db)                       # creates schema + sets the norm_v marker
    g.db.execute("DELETE FROM kg_state WHERE k='norm_v'")      # pretend: pre-fix graph
    g.db.execute("INSERT INTO kg_nodes(id,type,label,norm,aliases,mentions,first_ts,"
                 "last_ts,status,locked) VALUES('person:reykjav k','person','Reykjavík',"
                 "'reykjav k','[]',3,1,1,'active',0)")
    old_eid = mg._edge_key(mg.USER_ID, "visited", "person:reykjav k")
    g.db.execute("INSERT INTO kg_edges(id,src,dst,rel,mentions,first_ts,last_ts,"
                 "valid_from,valid_to,source_turn,quote,fact,status) VALUES(?,?,?,?,2,"
                 "1,1,1,NULL,1,'q','visited Reykjavík','active')",
                 (old_eid, mg.USER_ID, "person:reykjav k", "visited"))
    g.db.commit()
    g2 = mg.MindGraph(db)                      # re-open → migration runs
    row = g2.db.execute("SELECT id, norm, mentions FROM kg_nodes "
                        "WHERE id='person:reykjavík'").fetchone()
    check("the mangled node key is renamed in place (mentions kept)",
          row is not None and row[1] == "reykjavík" and row[2] == 3)
    check("no ghost stays under the old key",
          g2.db.execute("SELECT COUNT(*) FROM kg_nodes WHERE id='person:reykjav k'")
             .fetchone()[0] == 0)
    ne = mg._edge_key(mg.USER_ID, "visited", "person:reykjavík")
    check("the edge follows: endpoints AND primary key re-keyed",
          g2.db.execute("SELECT dst FROM kg_edges WHERE id=?", (ne,)).fetchone()
          == ("person:reykjavík",))
    check("migration is one-time (marker set)",
          g2.db.execute("SELECT v FROM kg_state WHERE k='norm_v'").fetchone() == ("2",))


def test_batch_char_budget():
    """The 'stuck at 1020' bug, one level up from the poison turn: per-TURN clipping
    bounded each turn at 4000 chars, but a batch of 40 long-form turns still stacked to
    ~160k chars — an extraction call that times out on EVERY pass, so the backlog froze
    (and moved exactly one half-bisect when a lucky sub-batch fit).  Batches are now
    bounded by total chars too, so heavy stretches of transcript cost roughly the same
    call as one-liners."""
    db = fresh_db()
    for i in range(10):
        add_turn(db, "user", f"[{i}] " + ("long-form medicine talk " * 40))   # ~1k chars each
    g = mg.MindGraph(db, {"distill_batch_turns": 8, "distill_batch_chars": 3000})
    turns = g._new_user_turns(8)
    check("a batch stops accumulating at the char budget (not 8 long turns)",
          1 <= len(turns) <= 3)
    prompts = []

    def fn(prompt):
        prompts.append(prompt)
        return {}
    st = asyncio.run(g.catch_up_all(fn))
    check("the whole backlog still drains, in more + smaller batches",
          st["done"] and g.backlog() == 0 and len(prompts) >= 4)
    tmpl = len(g.build_prompt([]))                     # the fixed instruction template
    check("every extraction prompt's BODY honoured the budget",
          all(len(p) - tmpl < 3000 + 300 for p in prompts))

    # one turn larger than the whole budget still goes through (clipped), alone
    db2 = fresh_db()
    add_turn(db2, "user", "z" * 9000)
    add_turn(db2, "user", "short one")
    g2 = mg.MindGraph(db2, {"distill_batch_turns": 8, "distill_batch_chars": 3000,
                            "max_turn_chars": 4000})
    t1 = g2._new_user_turns(8)
    check("an over-budget turn is taken ALONE, never starved",
          len(t1) == 1 and t1[0]["text"].startswith("z"))
    asyncio.run(g2.distill(stub({})))
    t2 = g2._new_user_turns(8)
    check("…and the next batch resumes right after it",
          len(t2) == 1 and t2[0]["text"] == "short one")
    g3 = mg.MindGraph(fresh_db(), {"distill_batch_chars": 0})
    check("0 disables the char budget (count-only batching preserved)",
          g3.c["distill_batch_chars"] == 0)


def main():
    test_anchor_and_schema()
    test_prompt_carries_a_grounded_worked_example()
    test_paraphrased_fact_is_surfaced_not_the_snippet()
    test_fact_absent_falls_back_to_triple()
    test_corroboration_updates_the_fact()
    test_failed_extraction_does_not_advance()
    test_catch_up_stops_on_failure()
    test_catch_up_all_drains_past_the_one_pass_ceiling()
    test_poison_turn_cannot_wedge_the_backlog()
    test_value_coming_back_resurrects_not_crashes()
    test_retracted_edge_resurrects_on_fresh_evidence()
    test_fold_crash_cannot_wedge_the_backlog()
    test_non_latin_labels()
    test_norm_migration_renames_mangled_keys()
    test_batch_char_budget()
    test_grounded_fold()
    test_grounding_refuses_invention()
    test_identity_firewall()
    test_cardinality_supersession()
    test_corroboration_and_checkpoint()
    test_reversible_retract()
    test_deterministic_fold()
    test_snapshot()
    test_no_duplicate_distilling()
    test_catch_up_backfills_old_chats()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
