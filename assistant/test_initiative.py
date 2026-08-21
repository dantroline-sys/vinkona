#!/usr/bin/env python
"""VIN-INIT-01 deterministic core (initiative.py) — IN1+IN2.

The §10 criteria testable offline: grounding is mandatory; a planted open-loop
item is raised inside its window, never before, never after expiry; a
single-source news item is not speakable and a corroborated one is;
grief-adjacent never initiates; two deflections retire; never twice in one
conversation (unless engaged + invited); provenance answers "why did you ask";
20 simulated greetings hold the frequency gate and invitations always bypass.

    python assistant/test_initiative.py
"""
import sqlite3

import initiative
from initiative import InitiativeQueue, feed_backlog, feed_self_state, is_invitation

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}" + (f" — {detail}" if detail else ""))


def q(**cfg):
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    return InitiativeQueue(db, cfg or None)


T0 = 1_000_000.0
DAY = 86_400.0


def test_grounding_mandatory():
    iq = q()
    r = iq.add("backlog", "an ungrounded thought", grounding=[], now=T0)
    check("an ungrounded item is refused",
          not r["ok"] and "confabulated" in r["error"])
    r = iq.add("backlog", "a grounded thought", grounding=["plan_question:1"],
               now=T0)
    check("a grounded item is admitted", r["ok"])
    check("provenance answers 'why did you ask me that?'",
          iq.provenance(r["item"]["id"]) == ["plan_question:1"])


def test_open_loop_window():
    iq = q()
    # The user's dreaded Thursday: window opens T0+2d, closes T0+4d.
    r = iq.add("open_loop", "how did the Thursday appointment go",
               grounding=["chat:123"], window=(T0 + 2 * DAY, T0 + 4 * DAY),
               now=T0, fit=0.5)
    iid = r["item"]["id"]
    check("before its window the item is not raised",
          iq.pick(T0 + DAY, invitation=True) is None)
    got = iq.pick(T0 + 3 * DAY, invitation=True)
    check("inside its window it is raised", got and got["id"] == iid)
    check("expiry is strict: window close + 7 days and it is GONE",
          iq.pick(T0 + 4 * DAY + initiative.OPEN_LOOP_GRACE_S + 1,
                  invitation=True) is None
          and iq.get(iid) is None)


def test_news_tact():
    iq = q()
    thin = iq.add("news", "a serious accident on the ring road",
                  grounding=["news:1"], corroborated=False, now=T0, fit=0.8)
    check("a single-source news item is stored but NOT speakable",
          thin["ok"] and iq.pick(T0 + 60, invitation=True) is None)
    iq.add("news", "flooding closed the valley line this morning",
           grounding=["news:2", "news:3"], corroborated=True, now=T0, fit=0.8)
    got = iq.pick(T0 + 60, invitation=True)
    check("a corroborated item becomes speakable",
          got and "flooding" in got["pointer"])


def test_grief_adjacent_never_initiates():
    iq = q()
    r = iq.add("news", "the obituary that mentions her colleague",
               grounding=["news:9"], sensitivity="grief_adjacent",
               corroborated=True, now=T0, fit=1.0)
    check("grief-adjacent items are forced respond-only",
          r["ok"] and r["item"]["respond_only"] is True)
    check("…and are never selected as openers",
          iq.pick(T0 + 60, invitation=True) is None)
    check("…but stay available for recall when the USER raises it",
          any(i["sensitivity"] == "grief_adjacent" for i in iq.items(T0 + 60)))


def test_fatigue_and_retirement():
    iq = q()
    r = iq.add("backlog", "the tidal-lock question", grounding=["pq:7"],
               now=T0, fit=0.6)
    iid = r["item"]["id"]
    iq.record_outcome(iid, "deflected")
    still = iq.pick(T0 + 60, invitation=True)
    check("one deflection discounts but may survive",
          iq.get(iid)["deflections"] == 1
          and (still is None or still["id"] == iid))
    iq.record_outcome(iid, "deflected")
    check("two deflections retire the item",
          iq.get(iid)["status"] == "retired"
          and iq.pick(T0 + 60, invitation=True) is None)

    r2 = iq.add("backlog", "the harbour-bridge question", grounding=["pq:8"],
                now=T0, fit=0.6)
    iq.record_outcome(r2["item"]["id"], "engaged")
    check("an engaged item retires with honour (its job is done)",
          iq.get(r2["item"]["id"])["status"] == "retired"
          and iq.get(r2["item"]["id"])["last_outcome"] == "engaged")

    r3 = iq.add("backlog", "the glacier question", grounding=["pq:9"],
                now=T0, fit=0.6)
    before = iq.salience(iq.get(r3["item"]["id"]), T0 + 60)
    iq.record_outcome(r3["item"]["id"], "dropped")
    after = iq.salience(iq.get(r3["item"]["id"]), T0 + 60)
    check("a model drop returns the item UNDISCOUNTED", before == after
          and iq.get(r3["item"]["id"])["raised_count"] == 0)


def test_session_governor():
    iq = q()
    iq.add("backlog", "the mushroom question", grounding=["pq:1"], now=T0,
           fit=0.6)
    check("after one raise this conversation, silence — unless invited",
          iq.pick(T0 + 60, invitation=False, session_raised=True,
                  rng=lambda: 0.0) is None)
    check("…an explicit invitation re-opens the door",
          iq.pick(T0 + 60, invitation=True, session_raised=True) is not None)


def test_frequency_gate_simulation():
    iq = q()
    iq.add("backlog", "the perennial question", grounding=["pq:2"], now=T0,
           fit=0.8)
    # a deterministic 20-greeting parade: rolls 0.025, 0.075, … 0.975
    rolls = [(i + 0.5) / 20 for i in range(20)]
    fired = sum(1 for r in rolls
                if iq.pick(T0 + 60, invitation=False, p=0.6,
                           rng=lambda r=r: r) is not None)
    check(f"across 20 greetings the rate matches p ({fired}/20 at p=0.6)",
          fired == 12)
    check("explicit invitations always bypass the gate",
          all(iq.pick(T0 + 60, invitation=True) is not None for _ in range(5)))
    check("the invitation phrase list recognises the classic opener",
          is_invitation("So, what's new in your world?")
          and not is_invitation("can you set a timer for ten minutes"))


def test_queue_hygiene():
    iq = q(max_queue=3)
    ptrs = ["the mushroom soup question", "the harbour tides question",
            "the violin bow question", "the glacier melt question"]
    for i, p in enumerate(ptrs):
        iq.add("backlog", p, grounding=[f"pq:{i}"], now=T0, fit=0.2 * i)
    live = iq.items(T0 + 60)
    check("the cap holds and the weakest item was pruned",
          len(live) == 3 and all("mushroom" not in i["pointer"] for i in live))
    dup = iq.add("backlog", "the glacier melt question",
                 grounding=["pq:99"], now=T0)
    check("a near-identical pointer is deduped", dup.get("duplicate") is True)
    iq.clear()
    check("clear empties the queue (privacy tab)", iq.items(T0) == [])


def test_novelty_channel_penalty():
    iq = q()
    iq.add("news", "storm damage on the coast", grounding=["n:1", "n:2"],
           corroborated=True, now=T0, fit=0.5)
    iq.add("backlog", "the lighthouse question", grounding=["pq:5"], now=T0,
           fit=0.5)
    got = iq.pick(T0 + 60, invitation=True, last_channel="news")
    check("two news openers in a row are penalised — variety wins",
          got and got["channel"] == "backlog")


def test_feeders():
    iq = q()
    n = feed_self_state(iq, [
        {"kind": "toolsmith", "action": "deployed", "name": "rain_news"},
        {"kind": "toolsmith", "action": "failed", "title": "x"},   # not news
        {"kind": "graph_run", "ok": True, "name": "rain_news", "forced": False},
        {"kind": "garden", "pruned": 3},                            # unmapped
    ], now=T0)
    check("self-state feeder keeps the deployments, skips the noise", n == 2)
    check("…and the pointers are first-person and honest",
          any("new tool for myself" in i["pointer"] for i in iq.items(T0 + 1)))

    n = feed_backlog(iq, [
        {"id": 7, "question": "why do tidal locks stabilise", "topic": "orbital mechanics"},
        {"id": 8, "question": "", "topic": "empty"},
    ], graph_terms={"tidal", "orbital", "sailing"}, now=T0)
    check("backlog feeder grounds items in their plan question", n == 1
          and any(i["grounding"] == ["plan_question:7"] for i in iq.items(T0 + 1)))
    it = next(i for i in iq.items(T0 + 1) if i["channel"] == "backlog")
    check("graph-term overlap lands as relational fit", it["fit"] > 0)


def test_opener_lane():
    import time as _time
    now = _time.time()                    # the lane lives on the wall clock
    iq = q()
    lane = initiative.OpenerLane(iq, {"enabled": True, "p_open": 1.0})

    blk = lane.block("so, what's new?", session_id="s1")
    check("empty queue + invitation → the honesty block, never a volley",
          "never turn the question back" in blk and "invent" in blk)

    iq.add("backlog", "the falcon migration question I left open",
           grounding=["plan_question:3"], fit=0.8, now=now)
    blk = lane.block("what's new with you?", session_id="s1")
    check("an item renders as ONE pointer with the drop-it-freely rule",
          "falcon migration" in blk and "drop it without a word" in blk
          and "channel" not in blk)

    lane.spoken("I keep coming back to the falcon migration question — "
                "did you ever look into it?")
    it = iq.items(now + 60)[0]
    check("the watermark records a real raise", it["raised_count"] == 1)
    lane.judge("oh the falcons! yes — don't they cross the strait in a day?")
    check("their take-up lands as engaged (item retires)",
          iq.get(it["id"])["status"] == "retired"
          and iq.get(it["id"])["last_outcome"] == "engaged")

    r2 = iq.add("backlog", "the glass harmonica repertoire question",
                grounding=["plan_question:4"], fit=0.8, now=now)
    blk = lane.block("morning", session_id="s2", rng=lambda: 0.0)
    check("a fresh session's first turn may carry an opener",
          "glass harmonica" in blk)
    lane.spoken("Morning! Sleep well?")
    check("an unspoken candidate returns UNDISCOUNTED",
          iq.get(r2["item"]["id"])["raised_count"] == 0)
    check("mid-session small talk carries nothing",
          lane.block("tell me about cheese", session_id="s2") == "")

    off = initiative.OpenerLane(iq, {"enabled": False})
    check("a disabled lane says nothing at all",
          off.block("what's new?", session_id="s3") == "")


def main():
    test_grounding_mandatory()
    test_open_loop_window()
    test_news_tact()
    test_grief_adjacent_never_initiates()
    test_fatigue_and_retirement()
    test_session_governor()
    test_frequency_gate_simulation()
    test_queue_hygiene()
    test_novelty_channel_penalty()
    test_feeders()
    test_opener_lane()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
