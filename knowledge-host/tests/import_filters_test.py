"""Tests for the per-dataset import filters — the user-tunable 'what do I care
about' knobs:

  * conceptnet: `exclude` relation names are ALWAYS skipped, even when
    include_lexical would admit them; unknown names warn (typo guard);
  * causenet: `min_sources` counts DISTINCT sources (same sentence scraped
    thrice = 1), and the corroboration number stored on the edge is distinct;
  * atomic: 'none' annotations never import; `min_count` is the annotator
    agreement floor.

Run:  python tests/import_filters_test.py     (stdlib only)
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost.kb import KB
from knowledgehost import atomic, causenet, conceptnet


def check(label, cond):
    print(("  ok  " if cond else "  FAIL ") + label)
    if not cond:
        check.failed += 1
check.failed = 0


def fresh_kb(td, name):
    return KB({"kb_path": os.path.join(td, name)})


def cn_line(rel, a, b, weight=2.0):
    return (f"/a/x\t/r/{rel}\t/c/en/{a}\t/c/en/{b}\t"
            + json.dumps({"weight": weight}) + "\n")


def edge_types(kb):
    return sorted(t for (t,) in kb.db.execute("SELECT type FROM edges").fetchall())


def main():
    td = tempfile.mkdtemp(prefix="kb-filters-")

    # ── conceptnet exclude ───────────────────────────────────────────────────
    cn = os.path.join(td, "assertions.csv")
    with open(cn, "w") as f:
        f.write(cn_line("IsA", "cat", "animal"))
        f.write(cn_line("Causes", "fire", "smoke"))
        f.write(cn_line("FormOf", "cats", "cat"))          # lexical
        f.write(cn_line("Synonym", "sofa", "couch"))       # lexical but NOT default-skipped

    kb = fresh_kb(td, "a.db")
    st = conceptnet.import_conceptnet(kb, cn)              # defaults: lexical set skipped
    check("default: FormOf skipped by the lexical toggle, Synonym kept",
          st["imported"] == 3 and "form_of" not in edge_types(kb)
          and "synonym" in edge_types(kb))
    kb.close()

    kb = fresh_kb(td, "b.db")
    st = conceptnet.import_conceptnet(kb, cn, include_lexical=True,
                                      exclude=["FormOf"])
    check("include_lexical + exclude: FormOf still skipped (exclude always wins)",
          st["imported"] == 3 and "form_of" not in edge_types(kb))
    kb.close()

    kb = fresh_kb(td, "c.db")
    st = conceptnet.import_conceptnet(kb, cn, exclude=["Causes", "Sinonym"])
    check("exclude drops a non-lexical relation too", "causes" not in edge_types(kb))
    check("typo'd exclude name only warns, import continues", st["imported"] == 2)
    kb.close()

    # ── causenet distinct-source floor ───────────────────────────────────────
    def rec(cause, effect, sources):
        return json.dumps({"causal_relation": {"cause": {"concept": cause},
                                               "effect": {"concept": effect}},
                           "sources": sources}) + "\n"

    def wiki(page, sent):
        return {"type": "wikipedia_sentence",
                "payload": {"sentence": sent, "wikipedia_page_title": page}}

    cnet = os.path.join(td, "causenet.jsonl")
    with open(cnet, "w") as f:
        # same sentence scraped three times -> 1 DISTINCT source
        f.write(rec("smoking", "cancer", [wiki("Smoking", "Smoking causes cancer.")] * 3))
        # two different pages -> 2 distinct
        f.write(rec("rain", "flood", [wiki("Rain", "Rain causes floods."),
                                      wiki("Flood", "Floods are caused by rain.")]))

    kb = fresh_kb(td, "d.db")
    st = causenet.import_causenet(kb, cnet, min_sources=2)
    check("min_sources=2: triple-scraped single source filtered, 2-page record kept",
          st["imported"] == 1 and st["skip_sources"] == 1)
    mods = json.loads(kb.db.execute("SELECT modifiers FROM edges").fetchall()[0][0])
    check("edge corroboration number is the DISTINCT count", mods["causenet_sources"] == 2)
    kb.close()

    kb = fresh_kb(td, "e.db")
    st = causenet.import_causenet(kb, cnet)                # default floor: keep all
    check("default keeps both", st["imported"] == 2)
    kb.close()

    # ── atomic: 'none' exclusion + agreement floor ───────────────────────────
    at = os.path.join(td, "v4_atomic_all_agg.csv")
    with open(at, "w", newline="") as f:
        import csv as _csv
        w = _csv.DictWriter(f, fieldnames=["event", "xIntent", "xReact"])
        w.writeheader()
        w.writerow({"event": "PersonX pays the bill",
                    "xIntent": json.dumps(["none", "to be polite", "To Be Polite",
                                           "to be polite"]),
                    "xReact": json.dumps(["generous"])})

    kb = fresh_kb(td, "f.db")
    st = atomic.import_atomic(kb, at, min_count=2)
    check("'none' annotations never import", st["skip_none"] == 1)
    check("min_count=2: 3x-agreed intent kept, single-annotator react dropped",
          st["imported"] == 1 and st["skip_count"] == 1)
    lbl = kb.db.execute("SELECT label FROM nodes WHERE kind='concept'").fetchall()
    check("agreed annotation imported once, case-collapsed",
          len(lbl) == 1 and lbl[0][0].lower() == "to be polite")
    kb.close()

    print(f"\n{'ALL OK' if not check.failed else str(check.failed) + ' FAILED'}")
    raise SystemExit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
