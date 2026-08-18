#!/usr/bin/env python
"""VIN-TOOL-01 wiring validator (toolgraph.py) — the §6.1 static pass.

The §11 criteria this file pins: a deliberately mis-wired graph is rejected
before execution; the capability summary renders in plain language; versions
pin.  Connection semantics deliberately mirror Haystack 2.x's pre-run socket
validation (§0.4): typed ports, checked before anything runs, legible errors.

    python assistant/test_toolgraph.py
"""
import palette  # noqa: F401 — populates the registry
import toolgraph
from toolgraph import capability_summary, needs_approval, validate

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}" + (f" — {detail}" if detail else ""))


def news_graph(**over):
    """The v0 use-case: feed → filter → dedupe → rank → head → digest."""
    g = {
        "name": "morning_news", "goal": "A short weather digest from my feed.",
        "steps": [
            {"id": "s1", "block": "rss_fetch",
             "inputs": {"feed": {"url": "http://a/feed"}}},
            {"id": "s2", "block": "filter_predicate",
             "params": {"terms": ["weather", "rain"]},
             "inputs": {"docs": "$s1.docs"}},
            {"id": "s3", "block": "dedupe", "inputs": {"docs": "$s2.docs"}},
            {"id": "s4", "block": "sort_rank",
             "inputs": {"docs": "$s3.docs", "query": {"text": "weather"}}},
            {"id": "s5", "block": "digest_render",
             "inputs": {"ranked": "$s4.ranked"}},
        ],
        "outputs": {"digest": "$s5.digest"},
    }
    g.update(over)
    return g


def test_valid_graph():
    v = validate(news_graph())
    check("the news graph validates", v.ok, "; ".join(v.errors))
    check("capability set computed before execution", v.capabilities == {"net"})
    check("steps come back pinned to exact versions",
          all(s["block_version"] == "1.0.0" for s in v.pinned["steps"]))
    check("defaults are filled into the pinned graph",
          v.pinned["steps"][0]["params"]["max_items"] == 30)
    order = [s["id"] for s in v.pinned["steps"]]
    check("pinned steps are in execution order",
          order.index("s1") < order.index("s2") < order.index("s4"))


def test_miswired_rejected():
    # List[Document] into a Ranked[Document] port: the §11 mis-wire.
    g = news_graph()
    g["steps"][4]["inputs"]["ranked"] = "$s3.docs"
    v = validate(g)
    check("List into a Ranked port is rejected", not v.ok)
    check("…and the error suggests an adapter",
          any("adapter" in e for e in v.errors), "; ".join(v.errors))
    # Ranked into List IS allowed (order safely forgotten): rank then re-filter.
    g2 = news_graph()
    g2["steps"].append({"id": "s6", "block": "filter_predicate",
                        "params": {"terms": ["rain"]},
                        "inputs": {"docs": "$s4.ranked"}})
    check("Ranked into a List port connects", validate(g2).ok,
          "; ".join(validate(g2).errors))


def test_bad_shapes_rejected():
    cases = [
        ("unknown block", {"steps": [{"id": "s1", "block": "quantum_leap",
                                      "inputs": {}}]}, "no block named"),
        ("version pin miss",
         {"steps": [{"id": "s1", "block": "rss_fetch", "block_version": "9.9.9",
                     "inputs": {"feed": {"url": "http://a"}}}]}, "not in registry"),
        ("unbound input", {"steps": [{"id": "s1", "block": "rss_fetch",
                                      "inputs": {}}]}, "unbound"),
        ("unknown port", {"steps": [{"id": "s1", "block": "rss_fetch",
                                     "inputs": {"feed": {"url": "http://a"},
                                                "nope": "$s1.docs"}}]},
         "no such input port"),
        ("bad literal shape", {"steps": [{"id": "s1", "block": "rss_fetch",
                                          "inputs": {"feed": {"link": "x"}}}]},
         "literal"),
        ("unknown param", {"steps": [{"id": "s1", "block": "rss_fetch",
                                      "params": {"turbo": 1},
                                      "inputs": {"feed": {"url": "http://a"}}}]},
         "unknown param"),
        ("dangling ref", {"steps": [{"id": "s1", "block": "dedupe",
                                     "inputs": {"docs": "$ghost.docs"}}]},
         "points at no step"),
        ("self-feed", {"steps": [{"id": "s1", "block": "dedupe",
                                  "inputs": {"docs": "$s1.docs"}}]}, "itself"),
        ("missing goal", {"goal": " "}, "goal"),
        ("bad output ref", {"outputs": {"digest": "$s9.digest"}},
         "points at nothing"),
    ]
    for label, over, needle in cases:
        g = news_graph(**over)
        v = validate(g)
        hit = (not v.ok) and any(needle in e for e in v.errors)
        check(f"rejects {label}", hit, "; ".join(v.errors) or "validated?!")


def test_cycle_rejected():
    g = {"name": "loop", "goal": "g",
         "steps": [{"id": "s1", "block": "dedupe", "inputs": {"docs": "$s2.docs"}},
                   {"id": "s2", "block": "dedupe", "inputs": {"docs": "$s1.docs"}}],
         "outputs": {"docs": "$s2.docs"}}
    v = validate(g)
    check("a cycle is rejected", not v.ok and any("cycle" in e for e in v.errors),
          "; ".join(v.errors))


def test_capability_language():
    v = validate(news_graph())
    check("net-only summary is honest",
          capability_summary(v) == "This tool can reach the internet.")
    check("a pure-transform graph says so",
          "touches nothing" in capability_summary(validate({
              "name": "trim", "goal": "top three",
              "steps": [{"id": "s1", "block": "ranked_head", "params": {"n": 3},
                         "inputs": {"ranked": []}}],
              "outputs": {"docs": "$s1.docs"}})))
    check("net alone needs no approval", not needs_approval(v))
    v.capabilities = {"net", "fs-write"}
    check("net+fs-write needs approval", needs_approval(v))
    v.capabilities = {"mutate"}
    check("mutate needs approval", needs_approval(v))
    v.capabilities = {"biometric"}
    check("biometric always needs approval", needs_approval(v))


def main():
    test_valid_graph()
    test_miswired_rejected()
    test_bad_shapes_rejected()
    test_cycle_rejected()
    test_capability_language()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
