#!/usr/bin/env python
"""VIN-TOOL-01 block layer (blocks.py + palette.py).

What must stay true, per self_tooling_spec.md:
  * the whole palette passes fixture CI, fast (§11);
  * the contract is ENFORCED at registration — bare types, missing fixtures,
    undeclared rollback, unknown capabilities all fail loudly (§3);
  * schemas are derived, never stored — no schema/grammar files exist (§11);
  * replay contexts fail on a miss instead of inventing data (§3);
  * the manifest detects a tampered block (§0.7);
  * the model-facing catalogue never contains source (§3).

    python assistant/test_blocks.py
"""
import time
from pathlib import Path

import blocks
import palette  # noqa: F401  — importing populates the registry
from blocks import Block, BlockError, ReplayCtx

HERE = Path(__file__).parent

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok  {name}")
    else:
        FAIL += 1; print(f"FAIL  {name}" + (f" — {detail}" if detail else ""))


# ── fixture CI first, before tests add dummy blocks to the registry ──────────
def test_fixture_ci():
    t0 = time.monotonic()
    failures, count = blocks.run_all_fixtures()
    dt = time.monotonic() - t0
    check(f"the whole palette passes fixture CI ({count} fixtures)", not failures,
          "; ".join(failures[:4]))
    check(f"fixture CI is fast ({dt:.2f}s)", dt < 60.0)
    check("palette v0 has the news core", {
        "rss_fetch", "filter_predicate", "dedupe", "sort_rank", "ranked_head",
        "summarise", "digest_render"}.issubset(set(blocks.names())))
    per_block = min(len(b.fixtures) for n in blocks.names()
                    for b in blocks._REGISTRY[n].values())
    check("every block ships at least 2 fixtures", per_block >= 2)


def test_types():
    check("Ranked[Document] parses", blocks.parse_type("Ranked[Document]") == ("Ranked", "Document"))
    for bad in ("dict", "list", "str", "List[dict]", "Set[Document]", "", "List[Document"):
        try:
            blocks.parse_type(bad)
            check(f"bare/foreign type {bad!r} rejected", False)
        except BlockError:
            check(f"bare/foreign type {bad!r} rejected", True)
    check("exact types connect", blocks.types_compatible("Query", "Query"))
    check("Ranked may feed List (order safely forgotten)",
          blocks.types_compatible("Ranked[Document]", "List[Document]"))
    check("List may NOT feed Ranked",
          not blocks.types_compatible("List[Document]", "Ranked[Document]"))
    check("different bases never connect",
          not blocks.types_compatible("List[Document]", "List[Passage]"))
    check("shape check catches a wrong item",
          blocks.check_value("List[Document]", [{"no_text": 1}]) is not None)
    check("shape check passes a digest", blocks.check_value("Digest", {"text": ""}) is None)


def _mk(**over):
    base = dict(
        name="dummy", version="1.0.0", summary="s",
        ports_in={"q": "Query"}, ports_out={"q": "Query"}, params={},
        capabilities=frozenset(), failure_modes=("none",),
        fixtures=({"name": "a", "inputs": {"q": {"text": "x"}},
                   "expect": {"q": {"text": "x"}}},
                  {"name": "b", "inputs": {"q": {"text": "y"}},
                   "expect": {"q": {"text": "y"}}}),
        fn=lambda i, p, c: {"q": i["q"]})
    base.update(over)
    return Block(**base)


def test_contract_enforced():
    cases = [
        ("bare dict port", dict(ports_in={"x": "dict"}), "closed semantic set"),
        ("one fixture only", dict(fixtures=(_mk().fixtures[0],)), "2 fixtures"),
        ("mutate without rollback", dict(capabilities=frozenset({"mutate"})), "rollback"),
        ("unknown capability", dict(capabilities=frozenset({"root"})), "unknown cap"),
        ("no failure modes", dict(failure_modes=()), "failure modes"),
        ("non-semver version", dict(version="1.0"), "semver"),
        ("param without description",
         dict(params={"p": {"type": "string"}}), "description"),
        ("no outputs", dict(ports_out={}), "produce"),
    ]
    for label, over, needle in cases:
        try:
            blocks.register(_mk(**over))
            check(f"registration rejects {label}", False)
        except BlockError as e:
            check(f"registration rejects {label}", needle.split()[0] in str(e).lower()
                  or needle in str(e), str(e))
    # and a valid mutate block WITH rollback is accepted
    ok = _mk(name="dummy_mut", capabilities=frozenset({"mutate"}),
             rollback="the tool store's notes table")
    blocks.register(ok)
    check("mutate with a declared rollback registers", blocks.get("dummy_mut") is ok)


def test_schema_derivation():
    b = blocks.get("filter_predicate")
    s = blocks.params_json_schema(b)
    check("derived schema forbids extras", s["additionalProperties"] is False)
    check("param without default is required", "terms" in s.get("required", []))
    check("enum and default survive derivation",
          s["properties"]["mode"]["enum"] == ["any", "all", "none"]
          and s["properties"]["mode"]["default"] == "any")
    check("array params carry item types",
          s["properties"]["terms"]["items"] == {"type": "string"})
    # §11: no schema or grammar files anywhere in the tree — schemas are derived.
    # Engine venvs (*_env) are third-party payload, not the repo's tree.
    stray = [p for pat in ("*.gbnf", "*schema*.json", "*.grammar")
             for p in HERE.rglob(pat)
             if not any(part.endswith("_env") for part in p.parts)]
    check("no schema/grammar files exist in the repo", not stray, str(stray))


def test_check_params():
    b = blocks.get("filter_predicate")
    check("unknown param flagged", blocks.check_params(b, {"terms": [], "x": 1}))
    check("missing required flagged",
          any("terms" in e for e in blocks.check_params(b, {})))
    check("enum miss flagged",
          any("mode" in e for e in blocks.check_params(b, {"terms": [], "mode": "some"})))
    rh = blocks.get("ranked_head")
    check("bool is not an integer",
          any("n" in e for e in blocks.check_params(rh, {"n": True})))
    check("good params pass", not blocks.check_params(b, {"terms": ["a"]}))


def test_replay_never_invents():
    try:
        ReplayCtx({}).net("http://never-canned")
        check("net replay miss fails loudly", False)
    except BlockError as e:
        check("net replay miss fails loudly", "replay miss" in str(e))
    try:
        ReplayCtx({}).llm("anything")
        check("llm replay miss fails loudly", False)
    except BlockError as e:
        check("llm replay miss fails loudly", "replay miss" in str(e))


def test_manifest():
    m1, m2 = blocks.manifest(), blocks.manifest()
    check("manifest is deterministic", m1 == m2 and m1)
    check("verify passes against itself", blocks.verify_manifest(m1) == [])
    tampered = dict(m1)
    key = sorted(tampered)[0]
    tampered[key] = "0" * 64
    check("verify names a tampered block", blocks.verify_manifest(tampered) == [key])


def test_catalogue_is_sourceless():
    cat = blocks.catalogue()
    check("catalogue covers the registry", len(cat) == len(blocks.names()))
    import json
    text = json.dumps(cat)
    check("the model-facing catalogue contains no source",
          "def " not in text and "lambda" not in text and "import " not in text)
    check("catalogue rows carry what emission needs",
          all({"name", "summary", "ports_in", "ports_out", "params",
               "capabilities"} <= set(r) for r in cat))


def main():
    test_fixture_ci()
    test_types()
    test_contract_enforced()
    test_schema_derivation()
    test_check_params()
    test_replay_never_invents()
    test_manifest()
    test_catalogue_is_sourceless()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
