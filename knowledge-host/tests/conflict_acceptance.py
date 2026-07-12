"""VINKONA-CONF-01 §11 acceptance tests — byte-exact conformance.

Builds the §11.1 ruleset (clinical + travel in one graph), runs CLIN / TRAV-1 / TRAV-2 /
TRAV-3, and compares the canonical output byte-for-byte against §11.3.  Per §11, the
harness rebuilds the ruleset, so ``ruleset_version`` is matched by ``^[0-9a-f]{64}$`` and
substituted into the expected strings; everything else must be byte-identical.

Run:  python tests/conflict_acceptance.py     (from the repo root; stdlib only)
"""
import json
import os
import re
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost.conflict import Checker, ConfError, canonical_json, ensure_schema

CAVEAT = ("no_known_conflicts means only that no ratified conflict rule fired for the "
          "checked actions against the provided state under closed-world predicate "
          "assumptions; it is NOT a safety determination. Unrepresented interactions, "
          "absent state, and unresolved quantities are not excluded.")

RS = "707a8e38853f0fca0f687b91485893879da5d82a50db39514121df1dd2ea7f17"  # spec's value; substituted

EXPECTED_CLIN = '{"checker_version":"1.0.0","ruleset_version":"' + RS + '","card_id":"card:clin.bronchospasm_mgmt","clearance":"conflicts_found","findings":[{"action":"act:administer_adrenaline","edge_id":"E1","relation_type":"contraindicated","disposition":"fire","reason":"triggered","severity":"severe","recommended_disposition":"warn_strong","mechanism":{"mechanism_id":"mech:unopposed_alpha","label":"Unopposed alpha-adrenergic effect","explanation":"Non-selective beta-blockade removes beta-2 vasodilation, leaving alpha-mediated vasoconstriction unopposed; risk of severe hypertension with reflex bradycardia.","conditionality_class":"acute_competition"},"rationale":"Sympathomimetic pressor response is altered by non-selective beta-blockade."},{"action":"act:administer_adrenaline","edge_id":"E4","relation_type":"contraindicated","disposition":"flag_for_human","reason":"indeterminate","severity":"caution","recommended_disposition":"human_review","mechanism":null,"rationale":"High-dose adrenaline caution in bronchospasm; dose-conditional."},{"action":"act:administer_adrenaline","edge_id":"E5","relation_type":"antagonizes","disposition":"flag_for_human","reason":"unratified_rule","severity":"caution","recommended_disposition":"human_review","mechanism":null,"rationale":"Possible exaggerated pressor response with MAOIs (proposed, unreviewed)."}],"checked":{"actions":["act:administer_adrenaline","act:administer_hydrocortisone"],"edges_consulted":["E1","E4","E5","E8"],"overrides_applied":[{"action":"act:administer_adrenaline","edge_id":"E8","override_id":"O3","justification":"Adrenaline age-caution handled by dedicated dosing guidance; generic advisory suppressed."}]},"coverage":{"caveat":"' + CAVEAT + '","not_evaluated":[{"edge_id":"E4","reason":"indeterminate_condition"},{"edge_id":"E5","reason":"unratified_rule"}]}}'

EXPECTED_TRAV1 = '{"checker_version":"1.0.0","ruleset_version":"' + RS + '","card_id":"card:trav.mel_lhr","clearance":"conflicts_found","findings":[{"action":"act:board_intl_flight","edge_id":"E2","relation_type":"requires","disposition":"fire","reason":"triggered","severity":"prohibitive","recommended_disposition":"block","mechanism":{"mechanism_id":"mech:carrier_boarding_rule","label":"Carrier boarding validity rule","explanation":"Carriers deny boarding when document validity at travel date is below the destination\'s required margin.","conditionality_class":"threshold"},"rationale":"Schengen entry requires at least 3 months document validity beyond travel date."}],"checked":{"actions":["act:board_intl_flight"],"edges_consulted":["E2"],"overrides_applied":[]},"coverage":{"caveat":"' + CAVEAT + '","not_evaluated":[]}}'

EXPECTED_TRAV2 = '{"checker_version":"1.0.0","ruleset_version":"' + RS + '","card_id":"card:trav.mel_lhr","clearance":"no_known_conflicts","findings":[],"checked":{"actions":["act:board_intl_flight"],"edges_consulted":["E2"],"overrides_applied":[]},"coverage":{"caveat":"' + CAVEAT + '","not_evaluated":[]}}'

EXPECTED_TRAV3 = '{"checker_version":"1.0.0","ruleset_version":"' + RS + '","card_id":"card:trav.mel_lhr","clearance":"review_required","findings":[{"action":"act:board_intl_flight","edge_id":"E2","relation_type":"requires","disposition":"flag_for_human","reason":"indeterminate","severity":"prohibitive","recommended_disposition":"human_review","mechanism":{"mechanism_id":"mech:carrier_boarding_rule","label":"Carrier boarding validity rule","explanation":"Carriers deny boarding when document validity at travel date is below the destination\'s required margin.","conditionality_class":"threshold"},"rationale":"Schengen entry requires at least 3 months document validity beyond travel date."}],"checked":{"actions":["act:board_intl_flight"],"edges_consulted":["E2"],"overrides_applied":[]},"coverage":{"caveat":"' + CAVEAT + '","not_evaluated":[{"edge_id":"E2","reason":"indeterminate_condition"}]}}'


def j(expr) -> str:
    return json.dumps(expr, separators=(",", ":"))


def build_ruleset(path: str) -> None:
    conn = sqlite3.connect(path)
    ensure_schema(conn)
    nodes = [
        ("act:administer_adrenaline", "administer adrenaline", "action"),
        ("act:administer_sympathomimetic", "administer a sympathomimetic", "class"),
        ("act:administer_hydrocortisone", "administer hydrocortisone", "action"),
        ("act:administer_steroid", "administer a steroid", "class"),
        ("act:board_intl_flight", "board an international flight", "action"),
        ("state:refractory_bronchospasm", "refractory bronchospasm", "predicate"),
        ("state:on_nonselective_beta_blocker", "on a non-selective beta-blocker", "predicate"),
        ("state:on_maoi", "on an MAOI", "predicate"),
        ("state:elderly", "elderly", "predicate"),
        ("dest:schengen", "Schengen-area destination", "predicate"),
    ]
    conn.executemany("INSERT INTO conflict_node(node_id,label,kind) VALUES(?,?,?)", nodes)
    conn.executemany("INSERT INTO conflict_is_a(child,parent,status) VALUES(?,?,'ratified')", [
        ("act:administer_adrenaline", "act:administer_sympathomimetic"),
        ("act:administer_hydrocortisone", "act:administer_steroid"),
    ])
    conn.executemany(
        "INSERT INTO mechanism(mechanism_id,label,explanation,conditionality_class) VALUES(?,?,?,?)", [
            ("mech:unopposed_alpha", "Unopposed alpha-adrenergic effect",
             "Non-selective beta-blockade removes beta-2 vasodilation, leaving alpha-mediated "
             "vasoconstriction unopposed; risk of severe hypertension with reflex bradycardia.",
             "acute_competition"),
            ("mech:carrier_boarding_rule", "Carrier boarding validity rule",
             "Carriers deny boarding when document validity at travel date is below the "
             "destination's required margin.", "threshold"),
        ])
    edges = [
        ("E1", "act:administer_sympathomimetic", "contraindicated", "severe",
         j({"op": "presence", "pred": "state:on_nonselective_beta_blocker"}),
         "mech:unopposed_alpha", "ratified", "pub",
         "Sympathomimetic pressor response is altered by non-selective beta-blockade.", None),
        ("E4", "act:administer_adrenaline", "contraindicated", "caution",
         j({"op": "all_of", "args": [
             {"op": "presence", "pred": "state:refractory_bronchospasm"},
             {"op": "compare", "field": "adrenaline_dose_mcg", "cmp": ">",
              "operand": {"lit": 500}}]}),
         None, "ratified", "pub",
         "High-dose adrenaline caution in bronchospasm; dose-conditional.", None),
        ("E5", "act:administer_sympathomimetic", "antagonizes", "caution",
         j({"op": "presence", "pred": "state:on_maoi"}),
         None, "proposed", "pub",
         "Possible exaggerated pressor response with MAOIs (proposed, unreviewed).", None),
        ("E8", "act:administer_sympathomimetic", "contraindicated", "advisory",
         j({"op": "presence", "pred": "state:elderly"}),
         None, "ratified", "pub",
         "Advisory caution for sympathomimetics in the elderly.", None),
        ("E2", "act:board_intl_flight", "requires", "prohibitive",
         j({"op": "all_of", "args": [
             {"op": "presence", "pred": "dest:schengen"},
             {"op": "compare", "field": "passport_validity_months_at_travel", "cmp": "<",
              "operand": {"lit": 3}}]}),
         "mech:carrier_boarding_rule", "ratified", "pub",
         "Schengen entry requires at least 3 months document validity beyond travel date.", None),
    ]
    conn.executemany(
        "INSERT INTO conflict_edge(edge_id,subject,relation_type,severity,fire_when,"
        "mechanism_id,status,authority,rationale,source_ref) VALUES(?,?,?,?,?,?,?,?,?,?)", edges)
    conn.execute(
        "INSERT INTO conflict_override(override_id,on_node,targets_edge_id,status,justification,"
        "source_ref) VALUES(?,?,?,?,?,?)",
        ("O3", "act:administer_adrenaline", "E8", "ratified",
         "Adrenaline age-caution handled by dedicated dosing guidance; generic advisory suppressed.",
         None))
    conn.commit()
    conn.close()


def run_case(checker, name, card, state, expected):
    out = canonical_json(checker.check(card, state))
    want = expected.replace(RS, checker.ruleset_version)
    if out != want:
        print(f"FAIL {name}")
        print("  got : " + out)
        print("  want: " + want)
        # first divergence, for fast diagnosis
        for i, (a, b) in enumerate(zip(out, want)):
            if a != b:
                print(f"  first divergence at byte {i}: got {out[i:i+60]!r} want {want[i:i+60]!r}")
                break
        sys.exit(1)
    print(f"PASS {name}  ({len(out)} bytes, byte-exact)")


def main():
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "conf.db")
        build_ruleset(db)
        checker = Checker.load(db)

        assert re.fullmatch(r"[0-9a-f]{64}", checker.ruleset_version), "ruleset_version format"
        print(f"ruleset_version = {checker.ruleset_version}  (harness-rebuilt; format-matched per §11)")

        clin_card = {"card_id": "card:clin.bronchospasm_mgmt",
                     "actions": ["act:administer_adrenaline", "act:administer_hydrocortisone"]}
        clin_state = {"predicates": ["state:refractory_bronchospasm",
                                     "state:on_nonselective_beta_blocker",
                                     "state:on_maoi", "state:elderly"], "fields": {}}
        trav_card = {"card_id": "card:trav.mel_lhr", "actions": ["act:board_intl_flight"]}

        run_case(checker, "CLIN  ", clin_card, clin_state, EXPECTED_CLIN)
        run_case(checker, "TRAV-1", trav_card,
                 {"predicates": ["dest:schengen"],
                  "fields": {"passport_validity_months_at_travel": 2.47}}, EXPECTED_TRAV1)
        run_case(checker, "TRAV-2", trav_card,
                 {"predicates": ["dest:schengen"],
                  "fields": {"passport_validity_months_at_travel": 9.0}}, EXPECTED_TRAV2)
        run_case(checker, "TRAV-3", trav_card,
                 {"predicates": ["dest:schengen"], "fields": {}}, EXPECTED_TRAV3)

        # ── determinism (§9): repeat run and a fresh load are byte-identical ──
        a = canonical_json(checker.check(clin_card, clin_state))
        b = canonical_json(Checker.load(db).check(clin_card, clin_state))
        assert a == b, "determinism across runs/loads"
        print("PASS determinism (repeat check + fresh load byte-identical)")

        # ── §9 (1.1 erratum): the scope-STRUCTURAL tables are hashed too ──────
        # is_a / administers / member_of / acts_via all change what fires, so
        # any row change must move ruleset_version — including a PROPOSED is_a
        # row that changes no behaviour (the audit trail must move even when
        # firing doesn't).  Reverting must restore the version exactly.
        base = checker.ruleset_version
        for table, ins, undo in [
            ("conflict_is_a (proposed)",
             "INSERT INTO conflict_is_a(child,parent,status) VALUES("
             "'act:administer_adrenaline','act:administer_steroid','proposed')",
             "DELETE FROM conflict_is_a WHERE parent='act:administer_steroid' "
             "AND child='act:administer_adrenaline'"),
            ("administers",
             "INSERT INTO administers(action,substance) VALUES("
             "'act:administer_adrenaline','act:administer_sympathomimetic')",
             "DELETE FROM administers"),
            ("member_of",
             "INSERT INTO member_of(child,grouper,grouper_type) VALUES("
             "'act:administer_adrenaline','act:administer_sympathomimetic','ATC')",
             "DELETE FROM member_of"),
            ("acts_via",
             "INSERT INTO acts_via(substance,mechanism,role) VALUES("
             "'act:administer_adrenaline','act:administer_sympathomimetic','moa')",
             "DELETE FROM acts_via"),
        ]:
            conn = sqlite3.connect(db); conn.execute(ins); conn.commit(); conn.close()
            moved = Checker.load(db).ruleset_version
            assert moved != base, f"{table}: row change did not move ruleset_version (the 1.0 audit gap)"
            conn = sqlite3.connect(db); conn.execute(undo); conn.commit(); conn.close()
            restored = Checker.load(db).ruleset_version
            assert restored == base, f"{table}: revert did not restore ruleset_version"
        print("PASS §9/1.1 scope-structural rows move ruleset_version (revert restores it)")

        # ── §10 error contract ──
        for bad_card, bad_state, code in [
            ({"card_id": "c", "actions": ["act:nonexistent"]},
             {"predicates": [], "fields": {}}, "E_UNKNOWN_NODE"),
            ({"card_id": "c", "actions": ["act:board_intl_flight"]},
             {"predicates": "oops", "fields": {}}, "E_MALFORMED_STATE"),
            ({"card_id": "c", "actions": ["act:board_intl_flight"]},
             {"predicates": [], "fields": {"x": True}}, "E_MALFORMED_STATE"),
            ({"card_id": "c", "actions": "act:board_intl_flight"},
             {"predicates": [], "fields": {}}, "E_MALFORMED_STATE"),
        ]:
            try:
                checker.check(bad_card, bad_state)
                print(f"FAIL error contract: expected {code}"); sys.exit(1)
            except ConfError as e:
                assert e.code == code, f"expected {code}, got {e.code}"
        print("PASS §10 error contract (E_UNKNOWN_NODE / E_MALFORMED_STATE)")

        # depth>4 expression must be rejected at LOAD (fail loud, never silently skipped)
        db2 = os.path.join(td, "bad.db")
        conn = sqlite3.connect(db2); ensure_schema(conn)
        conn.execute("INSERT INTO conflict_node(node_id) VALUES('act:x')")
        deep = {"op": "not", "arg": {"op": "not", "arg": {"op": "not", "arg": {
                "op": "not", "arg": {"op": "presence", "pred": "p"}}}}}
        conn.execute("INSERT INTO conflict_edge VALUES('B1','act:x','requires','caution',?,"
                     "NULL,'ratified','pub','r',NULL)", (json.dumps(deep),))
        conn.commit(); conn.close()
        try:
            Checker.load(db2)
            print("FAIL: depth-5 expression accepted"); sys.exit(1)
        except ConfError as e:
            assert e.code == "E_BAD_EXPRESSION"
        print("PASS E_BAD_EXPRESSION at load (depth bound enforced)")

        # a PROPOSED override must not suppress (spec §7 K3)
        db3 = os.path.join(td, "prop.db")
        conn = sqlite3.connect(db3); ensure_schema(conn)
        conn.executemany("INSERT INTO conflict_node(node_id) VALUES(?)", [("act:y",), ("state:p",)])
        conn.execute("INSERT INTO conflict_edge VALUES('X1','act:y','contraindicated','severe',?,"
                     "NULL,'ratified','pub','r',NULL)",
                     (j({"op": "presence", "pred": "state:p"}),))
        conn.execute("INSERT INTO conflict_override VALUES('OV1','act:y','X1','proposed','j',NULL)")
        conn.commit(); conn.close()
        out = Checker.load(db3).check({"card_id": "c", "actions": ["act:y"]},
                                      {"predicates": ["state:p"], "fields": {}})
        assert out["clearance"] == "conflicts_found" and not out["checked"]["overrides_applied"], \
            "proposed override must not suppress"
        print("PASS proposed override does not suppress")

        print("\nALL ACCEPTANCE TESTS PASS — VINKONA-CONF-01 §11 conformant")


if __name__ == "__main__":
    main()
