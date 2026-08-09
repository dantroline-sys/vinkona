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
          and all(k in e for e in ex["edges"] for k in ("src", "dst", "rel", "quote")))
    # the example must itself be GROUNDED (each quote occurs in the example input) — we teach
    # exactly the behaviour _fold enforces, so a model copying it produces edges that survive.
    low = mg.MindGraph._EXAMPLE_IN.lower()
    check("every example edge quote actually occurs in the example input (grounded)",
          all(e["quote"].lower() in low for e in ex["edges"]))


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


def main():
    test_anchor_and_schema()
    test_prompt_carries_a_grounded_worked_example()
    test_failed_extraction_does_not_advance()
    test_catch_up_stops_on_failure()
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
