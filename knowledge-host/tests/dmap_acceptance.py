"""VINKONA-DMAP-01 acceptance — the §7 worked example + end-to-end firing through CONF-01.

DMAP has no §11 byte-exact harness; its conformance target is:
  1. §7 worked example (adrenaline): correct ROUTING of each relation family, proposed
     status on the CI edge, provenance, and the exact node/edge shapes shown;
  2. §1 central rule negatively: has_MoA did NOT become a conflict, CI_with DID;
  3. §3 rules: R1 (proposed), R2 (subject=substance), R3 (provenance), R4 (severity),
     R5 (unresolved→research, not dropped), R6 (idempotent re-import);
  4. §5 ONC symmetric pair;
  5. §4 END-TO-END: the imported contraindication reaches act:administer_adrenaline via the
     `administers` scope link and FIRES once ratified — the whole reason the import exists.

Run:  python tests/dmap_acceptance.py     (from the repo root; stdlib only)
"""
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost import dmap
from knowledgehost.conflict import Checker

# resolver for the §7 adrenaline data: source string → canon node (crosswalk lives here)
RESOLVE = {
    ("substance", "adrenaline"): {"node_id": "substance:adrenaline", "label": "adrenaline"},
    ("class", "C01CA"): {"node_id": "class:atc.C01CA", "label": "Adrenergic and dopaminergic agents"},
    ("mechanism", "Adrenergic alpha-Agonists"):
        {"node_id": "mech:alpha_agonism", "label": "Adrenergic alpha-agonism"},
    ("mechanism", "Adrenergic beta-Agonists"):
        {"node_id": "mech:beta_agonism", "label": "Adrenergic beta-agonism"},
    ("condition", "Narrow-Angle Glaucoma"):
        {"node_id": "condition:narrow_angle_glaucoma", "label": "narrow-angle glaucoma"},
    ("condition", "Anaphylaxis"): {"node_id": "condition:anaphylaxis", "label": "anaphylaxis"},
}


def resolver(kind, name):
    return RESOLVE.get((kind, name))


RECORD = {
    "name": "adrenaline", "rxcui": "3992", "atc": ["C01CA24"], "version": "2026AA",
    "relations": [
        {"rela": "ATC", "source": "ATC", "target": "C01CA"},
        {"rela": "has_MoA", "source": "MED-RT", "target": "Adrenergic alpha-Agonists"},
        {"rela": "has_MoA", "source": "MED-RT", "target": "Adrenergic beta-Agonists"},
        {"rela": "CI_with", "source": "MED-RT", "target": "Narrow-Angle Glaucoma"},
        {"rela": "may_treat", "source": "MED-RT", "target": "Anaphylaxis"},
    ],
}


def edges_of(res, kind):
    return [e for e in res["edges"] if e["kind"] == kind]


def main():
    res = dmap.map_substance(RECORD, resolver)

    # ── §7 routing: each family in its lane ──
    member = edges_of(res, "member_of")
    acts = edges_of(res, "acts_via")
    confs = res["conflict_edges"]
    inds = edges_of(res, "indicated_for")

    assert len(member) == 1 and member[0]["to"] == "class:atc.C01CA" \
        and member[0]["grouper_type"] == "ATC", member
    assert {a["to"] for a in acts} == {"mech:alpha_agonism", "mech:beta_agonism"} \
        and all(a["role"] == "moa" for a in acts), acts
    assert len(confs) == 1, f"exactly one CI_with conflict edge, got {len(confs)}"
    print("PASS §7 routing (1 member_of, 2 acts_via, 1 conflict_edge, 1 indication)")

    ci = confs[0]
    assert ci["edge_id"] == "cirt:adrenaline:narrow_angle_glaucoma"
    assert ci["subject"] == "substance:adrenaline"                    # R2: subject is the substance
    assert ci["relation_type"] == "contraindicated"
    assert ci["fire_when"] == {"op": "presence", "pred": "condition:narrow_angle_glaucoma"}
    assert ci["status"] == "proposed"                                # R1: never authoritative on arrival
    assert ci["severity"] == "caution"                               # R4: MED-RT CI_with default
    assert ci["authority"] == "pub" and ci["source_ref"] == "MED-RT/CI_with/2026AA"  # R3
    assert ci["rationale"] == "MED-RT: adrenaline contraindicated with narrow-angle glaucoma"
    print("PASS §7 conflict edge (subject=substance, proposed, caution, MED-RT provenance)")

    # ── §1 negative: mechanism did NOT become a conflict; CI did ──
    assert not any(c["fire_when"].get("pred", "").startswith("mech:") for c in confs), \
        "has_MoA must not become a firing conflict"
    assert inds and inds[0]["to"] == "condition:anaphylaxis" \
        and inds[0]["from"] == "act:administer_adrenaline", inds
    print("PASS §1 central rule (has_MoA→substrate not conflict; may_treat→upstream indication)")

    # mechanism nodes carry conditionality_class from the §4 lookup (agonism→acute_competition)
    mechs = {m["mechanism_id"]: m for m in res["mechanisms"]}
    assert mechs["mech:alpha_agonism"]["conditionality_class"] == "acute_competition"
    print("PASS §4 mechanism conditionality_class inferred (agonism → acute_competition)")

    # ── R5: an unresolvable condition becomes a research item, not a dropped row ──
    rec2 = {"name": "adrenaline", "version": "2026AA",
            "relations": [{"rela": "CI_with", "source": "MED-RT", "target": "Unmapped Disease X"}]}
    res2 = dmap.map_substance(rec2, resolver)
    assert not res2["conflict_edges"]
    assert res2["research_items"] and res2["research_items"][0]["failed_side"] == "target" \
        and res2["research_items"][0]["target_str"] == "Unmapped Disease X" \
        and res2["research_items"][0]["type"] == "unresolved-mapping", res2["research_items"]
    print("PASS R5 unresolved condition → CONF-02 research item (never silently dropped)")

    # ── §5 ONC symmetric pair, both proposed + severe ──
    ddi = dmap.map_ddi_pair({"node_id": "substance:adrenaline", "label": "adrenaline"},
                            {"node_id": "substance:propranolol", "label": "propranolol"})
    assert len(ddi) == 2
    assert {e["edge_id"] for e in ddi} == {"onc:adrenaline:propranolol", "onc:propranolol:adrenaline"}
    assert ddi[0]["subject"] == "substance:adrenaline" \
        and ddi[0]["fire_when"] == {"op": "presence", "pred": "substance:propranolol"}
    assert all(e["status"] == "proposed" and e["severity"] == "severe" for e in ddi)  # R1, R4
    print("PASS §5 ONC DDI → symmetric proposed/severe pair (fires whichever drug is named)")

    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "kg.db")
        conn = sqlite3.connect(db)
        dmap.ensure_schema(conn)

        # persist the adrenaline import (idempotent: write twice, expect no duplicates)
        dmap.write(conn, res)
        dmap.write(conn, res)                                          # R6 idempotence
        n_ci = conn.execute("SELECT COUNT(*) FROM conflict_edge").fetchone()[0]
        assert n_ci == 1, f"re-import must be idempotent, got {n_ci} conflict edges"
        print("PASS R6 idempotent re-import (1 conflict edge after two writes)")

        # the authored-elsewhere action→substance link + the action node
        conn.execute("INSERT INTO conflict_node(node_id,label,kind) VALUES"
                     "('act:administer_adrenaline','administer adrenaline','action')")
        conn.execute("INSERT INTO administers(action,substance) VALUES"
                     "('act:administer_adrenaline','substance:adrenaline')")
        conn.commit()

        card = {"card_id": "card:anaphylaxis", "actions": ["act:administer_adrenaline"]}
        glaucoma = {"predicates": ["condition:narrow_angle_glaucoma"], "fields": {}}

        # PROPOSED (as imported): reaches the action via §4 scope, but only FLAGS — never fires.
        out = Checker.load(db).check(card, glaucoma)
        assert out["clearance"] == "review_required", out["clearance"]
        f = out["findings"][0]
        assert f["edge_id"] == "cirt:adrenaline:narrow_angle_glaucoma" \
            and f["disposition"] == "flag_for_human" and f["reason"] == "unratified_rule", f
        print("PASS §4 scope reaches action via `administers`; proposed edge FLAGS, not fires")

        # ratify it (the human step) → now it FIRES with the reviewer's severity.
        conn.execute("UPDATE conflict_edge SET status='ratified', severity='severe' "
                     "WHERE edge_id='cirt:adrenaline:narrow_angle_glaucoma'")
        conn.commit()
        out = Checker.load(db).check(card, glaucoma)
        assert out["clearance"] == "conflicts_found"
        f = out["findings"][0]
        assert f["disposition"] == "fire" and f["recommended_disposition"] == "warn_strong", f
        print("PASS §4 end-to-end: ratified contraindication FIRES on the administering action")

        # negative control: no glaucoma in state ⇒ no_known_conflicts (with the mandatory caveat)
        out = Checker.load(db).check(card, {"predicates": [], "fields": {}})
        assert out["clearance"] == "no_known_conflicts" and out["coverage"]["caveat"]
        print("PASS negative control (absent condition ⇒ no_known_conflicts + caveat)")

        # scope isolation: a DIFFERENT action that administers nothing sees no conflict
        conn.execute("INSERT INTO conflict_node(node_id,kind) VALUES('act:give_water','action')")
        conn.commit()
        out = Checker.load(db).check({"card_id": "c", "actions": ["act:give_water"]}, glaucoma)
        assert out["clearance"] == "no_known_conflicts"
        print("PASS scope isolation (non-administering action inherits no substance conflict)")

    print("\nALL ACCEPTANCE TESTS PASS — VINKONA-DMAP-01 §7 worked example + §4 end-to-end")


if __name__ == "__main__":
    main()
