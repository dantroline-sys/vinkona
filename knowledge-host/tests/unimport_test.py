"""Fidelity test for `unimport` — the threshold-tuning undo.

Imports two real datasets from tiny fixture files written in their actual dump
formats (ConceptNet assertions.csv lines, CauseNet precision JSONL), with a
deliberately FUSED node ("cancer" appears in both — ConceptNet creates it,
CauseNet's insert is ignored).  Then unimports one and verifies the surgical
properties that make tuning safe:

  * the dataset's edges are gone, the other dataset's remain;
  * a solely-dataset-supported node still referenced by the OTHER dataset's
    edges survives (shared structure is never orphaned);
  * unreferenced dataset-only nodes are deleted;
  * a mixed-support row (enriched later) survives with the dataset's support
    entries stripped;
  * the source_registry row is gone, and re-import restores everything;
  * unimporting the second dataset empties the graph.

Run:  python tests/unimport_test.py     (stdlib only)
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost.kb import KB
from knowledgehost import conceptnet, causenet, unimport as un


def check(label, cond):
    print(("  ok  " if cond else "  FAIL ") + label)
    if not cond:
        check.failed += 1
check.failed = 0


def cn_line(rel, a, b, weight=2.0):
    return (f"/a/x\t/r/{rel}\t/c/en/{a}\t/c/en/{b}\t" +
            json.dumps({"weight": weight}) + "\n")


def causenet_rec(cause, effect):
    return json.dumps({
        "causal_relation": {"cause": {"concept": cause}, "effect": {"concept": effect}},
        "sources": [{"type": "wikipedia_sentence",
                     "payload": {"sentence": f"{cause} causes {effect}.",
                                 "path_pattern": "", "sentence_type": "",
                                 "wikipedia_page_title": cause}}],
    }) + "\n"


def counts(kb):
    n = kb.db.execute("SELECT COUNT(*) FROM nodes").fetchall()[0][0]
    e = kb.db.execute("SELECT COUNT(*) FROM edges").fetchall()[0][0]
    s = kb.db.execute("SELECT COUNT(*) FROM source_registry").fetchall()[0][0]
    return n, e, s


def node_by_label(kb, label):
    r = kb.db.execute("SELECT id, support FROM nodes WHERE label=?", (label,)).fetchall()
    return r[0] if r else None


def main():
    td = tempfile.mkdtemp(prefix="kb-unimport-")
    cn_file = os.path.join(td, "assertions.csv")
    with open(cn_file, "w") as f:
        # smoking -> cancer -> death (conceptnet); "tumor" is conceptnet-only + unreferenced later
        f.write(cn_line("Causes", "smoking", "cancer"))
        f.write(cn_line("Causes", "cancer", "death"))
        f.write(cn_line("IsA", "tumor", "growth"))
    cnet_file = os.path.join(td, "causenet-precision.jsonl")
    with open(cnet_file, "w") as f:
        # cancer -> metastasis: 'cancer' FUSES with the conceptnet-created node
        f.write(causenet_rec("cancer", "metastasis"))

    kb = KB({"kb_path": os.path.join(td, "kb.db")})

    s1 = conceptnet.import_conceptnet(kb, cn_file, min_weight=1.0, trust=0.2)
    s2 = causenet.import_causenet(kb, cnet_file, trust=0.4, regime="conventional")
    check("fixture import: 3 cn edges + 1 causenet edge",
          s1["imported"] == 3 and s2["imported"] == 1)
    n0, e0, src0 = counts(kb)
    # labels: smoking cancer death tumor growth (cn) + metastasis (causenet),
    # with 'cancer' FUSED — 6 nodes, not 7
    check("fused node: 'cancer' exists once", n0 == 6 and e0 == 4 and src0 == 2)
    cancer = node_by_label(kb, "cancer")
    check("fusion left cancer solely conceptnet-supported",
          json.loads(cancer[1])[0]["doc_id"] == "conceptnet:5.7"
          and len(json.loads(cancer[1])) == 1)

    # enrich one conceptnet edge with a second (distilled) support entry
    eid = kb.db.execute("SELECT id, support FROM edges WHERE support LIKE '%conceptnet%' LIMIT 1").fetchall()[0]
    mixed = json.loads(eid[1]) + [{"doc_id": "doc:distilled-paper", "trust_weight": 0.8,
                                   "regime": "empirical", "has_reference": 1}]
    kb.db.execute("UPDATE edges SET support=? WHERE id=?", (json.dumps(mixed), eid[0]))
    kb.db.commit()

    # ── unimport conceptnet ──────────────────────────────────────────────────
    st = un.unimport(kb, un.DATASETS["conceptnet"])
    n1, e1, src1 = counts(kb)
    check("cn-only edges deleted (2 of 3; mixed one kept)",
          st["edges_deleted"] == 2 and st["edges_stripped"] == 1)
    check("mixed edge survives with cn support stripped",
          json.loads(kb.db.execute("SELECT support FROM edges WHERE id=?", (eid[0],)).fetchall()[0][0])
          [0]["doc_id"] == "doc:distilled-paper")
    cancer_after = node_by_label(kb, "cancer")
    check("fused 'cancer' survives (causenet edge references it)", cancer_after is not None)
    check("unreferenced cn-only nodes gone (tumor)", node_by_label(kb, "tumor") is None)
    check("conceptnet source row removed", src1 == 1)
    # the mixed edge's endpoints must also survive (referenced)
    check("mixed edge endpoints survive", e1 == 2 and n1 > 0)

    # ── re-import restores (the tuning loop) ────────────────────────────────
    s3 = conceptnet.import_conceptnet(kb, cn_file, min_weight=1.0, trust=0.2)
    n2, e2, src2 = counts(kb)
    check("re-import restores edges", e2 == 4 and src2 == 2 and s3["imported"] == 3)

    # ── unimport both: graph returns to (near) empty ─────────────────────────
    un.unimport(kb, un.DATASETS["conceptnet"])
    st4 = un.unimport(kb, un.DATASETS["causenet"])
    n3, e3, src3 = counts(kb)
    # the mixed edge (distilled support) and its two endpoints legitimately remain
    check("after both unimports only the distilled-support edge remains",
          e3 == 1 and src3 == 0 and n3 == 2)
    st5 = un.unimport(kb, un.DATASETS["causenet"])
    check("idempotent: second unimport is a no-op",
          st5["edges_deleted"] == 0 and st5["nodes_deleted"] == 0)

    kb.close()
    print(f"\n{'ALL OK' if not check.failed else str(check.failed) + ' FAILED'}")
    raise SystemExit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
