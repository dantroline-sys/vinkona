"""VINKONA-LEX-01 §11 acceptance tests — byte-exact conformance.

Compiles the §11.1 test lexicon against the canon registry, then verifies vectors
V1–V8 byte-for-byte against the §8.3 canonical serialization (lexicon_version matched
by ^[0-9a-f]{64}$ and substituted, per §11).  Also: the §4 tokenizer consequences, the
§6.3 compiler validation (C1–C4, EMPTY_NORM, warns; fail ⇒ report only), the §9 error
contract, and determinism across repeat matches and a fresh load.

Run:  python tests/lex_acceptance.py     (from the repo root; stdlib only)
"""
import json
import os
import re
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost.conflict import ensure_schema
from knowledgehost.lex import LexError, Matcher, compile_lexicon, norm_token, tokenize

US = "\x1f"
LV = "LVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLVLV"  # placeholder

ALIAS_DDL = """CREATE TABLE alias (
  alias_id      INTEGER PRIMARY KEY,
  node_id       TEXT    NOT NULL,
  surface       TEXT    NOT NULL,
  norm_seq      TEXT    NOT NULL,
  n_tokens      INTEGER NOT NULL CHECK (n_tokens BETWEEN 1 AND 8),
  alias_type    TEXT    NOT NULL CHECK (alias_type IN
                  ('preferred','synonym','index_term','variant','abbrev','informal','inflection')),
  weight        REAL    NOT NULL CHECK (weight > 0.0 AND weight <= 1.0),
  case_mode     TEXT    NOT NULL DEFAULT 'fold' CHECK (case_mode IN ('fold','exact','caps')),
  fuzzy_allowed INTEGER NOT NULL DEFAULT 1 CHECK (fuzzy_allowed IN (0, 1)),
  origin        TEXT    NOT NULL,
  derived_from  INTEGER REFERENCES alias(alias_id),
  status        TEXT    NOT NULL DEFAULT 'active' CHECK (status IN ('active','retired'))
)"""

LEXICON = [  # (alias_id, node_id, surface, alias_type, weight, case_mode, fuzzy_allowed)
    (1, "pub:proc.appendicectomy", "appendicectomy", "preferred", 1.00, "fold", 1),
    (2, "pub:proc.appendicectomy", "appendectomy", "variant", 0.85, "fold", 1),
    (3, "pub:dx.acute_appendicitis", "acute appendicitis", "preferred", 1.00, "fold", 1),
    (4, "pub:dx.appendicitis", "appendicitis", "preferred", 1.00, "fold", 1),
    (5, "pub:dx.pulm_embolism", "PE", "abbrev", 0.80, "caps", 0),
    (6, "pub:dx.pleural_effusion", "PE", "abbrev", 0.80, "caps", 0),
    (7, "pub:drug.hydralazine", "hydralazine", "preferred", 1.00, "fold", 0),
    (8, "pub:dx.parkinson_disease", "Parkinson disease", "preferred", 1.00, "fold", 1),
    (9, "pub:dx.ponv", "post op nausea", "informal", 0.70, "fold", 1),
]

V1 = ('{"matcher_version":"1.0.0","lexicon_version":"' + LV + '","norm_version":"NORM-1","tok_version":"TOK-1",'
      '"text":"Post-op nausea after appendicectomy.","tokens":['
      '{"i":0,"surface":"Post","norm":"post","char_start":0,"char_end":4,"corrected_from":null,"edit_distance":0},'
      '{"i":1,"surface":"op","norm":"op","char_start":5,"char_end":7,"corrected_from":null,"edit_distance":0},'
      '{"i":2,"surface":"nausea","norm":"nausea","char_start":8,"char_end":14,"corrected_from":null,"edit_distance":0},'
      '{"i":3,"surface":"after","norm":"after","char_start":15,"char_end":20,"corrected_from":null,"edit_distance":0},'
      '{"i":4,"surface":"appendicectomy","norm":"appendicectomy","char_start":21,"char_end":35,"corrected_from":null,"edit_distance":0}],'
      '"spans":['
      '{"span_id":0,"tok_start":0,"tok_end":3,"char_start":0,"char_end":14,"surface_original":"Post-op nausea",'
      '"matched_norm":"post op nausea","fuzzy":false,'
      '"candidates":[{"alias_id":9,"node_id":"pub:dx.ponv","alias_type":"informal","weight":0.7000,"score":0.7000}]},'
      '{"span_id":1,"tok_start":4,"tok_end":5,"char_start":21,"char_end":35,"surface_original":"appendicectomy",'
      '"matched_norm":"appendicectomy","fuzzy":false,'
      '"candidates":[{"alias_id":1,"node_id":"pub:proc.appendicectomy","alias_type":"preferred","weight":1.0000,"score":1.0000}]}],'
      '"unmatched_token_indices":[3],"flags":[]}')

V2 = ('{"matcher_version":"1.0.0","lexicon_version":"' + LV + '","norm_version":"NORM-1","tok_version":"TOK-1",'
      '"text":"Query PE.","tokens":['
      '{"i":0,"surface":"Query","norm":"query","char_start":0,"char_end":5,"corrected_from":null,"edit_distance":0},'
      '{"i":1,"surface":"PE","norm":"pe","char_start":6,"char_end":8,"corrected_from":null,"edit_distance":0}],'
      '"spans":[{"span_id":0,"tok_start":1,"tok_end":2,"char_start":6,"char_end":8,"surface_original":"PE",'
      '"matched_norm":"pe","fuzzy":false,"candidates":['
      '{"alias_id":5,"node_id":"pub:dx.pulm_embolism","alias_type":"abbrev","weight":0.8000,"score":0.8000},'
      '{"alias_id":6,"node_id":"pub:dx.pleural_effusion","alias_type":"abbrev","weight":0.8000,"score":0.8000}]}],'
      '"unmatched_token_indices":[0],"flags":[]}')

V3 = ('{"matcher_version":"1.0.0","lexicon_version":"' + LV + '","norm_version":"NORM-1","tok_version":"TOK-1",'
      '"text":"ruled out pe","tokens":['
      '{"i":0,"surface":"ruled","norm":"ruled","char_start":0,"char_end":5,"corrected_from":null,"edit_distance":0},'
      '{"i":1,"surface":"out","norm":"out","char_start":6,"char_end":9,"corrected_from":null,"edit_distance":0},'
      '{"i":2,"surface":"pe","norm":"pe","char_start":10,"char_end":12,"corrected_from":null,"edit_distance":0}],'
      '"spans":[],"unmatched_token_indices":[0,1,2],"flags":[]}')

V4 = ('{"matcher_version":"1.0.0","lexicon_version":"' + LV + '","norm_version":"NORM-1","tok_version":"TOK-1",'
      '"text":"urgent apendicectomy","tokens":['
      '{"i":0,"surface":"urgent","norm":"urgent","char_start":0,"char_end":6,"corrected_from":null,"edit_distance":0},'
      '{"i":1,"surface":"apendicectomy","norm":"appendicectomy","char_start":7,"char_end":20,'
      '"corrected_from":"apendicectomy","edit_distance":1}],'
      '"spans":[{"span_id":0,"tok_start":1,"tok_end":2,"char_start":7,"char_end":20,'
      '"surface_original":"apendicectomy","matched_norm":"appendicectomy","fuzzy":true,'
      '"candidates":[{"alias_id":1,"node_id":"pub:proc.appendicectomy","alias_type":"preferred","weight":1.0000,"score":0.8000}]}],'
      '"unmatched_token_indices":[0],"flags":[]}')

V5 = ('{"matcher_version":"1.0.0","lexicon_version":"' + LV + '","norm_version":"NORM-1","tok_version":"TOK-1",'
      '"text":"started hydralazne","tokens":['
      '{"i":0,"surface":"started","norm":"started","char_start":0,"char_end":7,"corrected_from":null,"edit_distance":0},'
      '{"i":1,"surface":"hydralazne","norm":"hydralazne","char_start":8,"char_end":18,"corrected_from":null,"edit_distance":0}],'
      '"spans":[],"unmatched_token_indices":[0,1],'
      '"flags":[{"type":"fuzzy_suppressed","stage":"token","token_index":1,"nearest":"hydralazine","distance":1}]}')

V6 = ('{"matcher_version":"1.0.0","lexicon_version":"' + LV + '","norm_version":"NORM-1","tok_version":"TOK-1",'
      '"text":"acute appendicitis confirmed","tokens":['
      '{"i":0,"surface":"acute","norm":"acute","char_start":0,"char_end":5,"corrected_from":null,"edit_distance":0},'
      '{"i":1,"surface":"appendicitis","norm":"appendicitis","char_start":6,"char_end":18,"corrected_from":null,"edit_distance":0},'
      '{"i":2,"surface":"confirmed","norm":"confirmed","char_start":19,"char_end":28,"corrected_from":null,"edit_distance":0}],'
      '"spans":[{"span_id":0,"tok_start":0,"tok_end":2,"char_start":0,"char_end":18,'
      '"surface_original":"acute appendicitis","matched_norm":"acute appendicitis","fuzzy":false,'
      '"candidates":[{"alias_id":3,"node_id":"pub:dx.acute_appendicitis","alias_type":"preferred","weight":1.0000,"score":1.0000}]}],'
      '"unmatched_token_indices":[2],"flags":[]}')

V7 = ('{"matcher_version":"1.0.0","lexicon_version":"' + LV + '","norm_version":"NORM-1","tok_version":"TOK-1",'
      '"text":"Parkinson\'s disease","tokens":['
      '{"i":0,"surface":"Parkinson\'s","norm":"parkinson","char_start":0,"char_end":11,"corrected_from":null,"edit_distance":0},'
      '{"i":1,"surface":"disease","norm":"disease","char_start":12,"char_end":19,"corrected_from":null,"edit_distance":0}],'
      '"spans":[{"span_id":0,"tok_start":0,"tok_end":2,"char_start":0,"char_end":19,'
      '"surface_original":"Parkinson\'s disease","matched_norm":"parkinson disease","fuzzy":false,'
      '"candidates":[{"alias_id":8,"node_id":"pub:dx.parkinson_disease","alias_type":"preferred","weight":1.0000,"score":1.0000}]}],'
      '"unmatched_token_indices":[],"flags":[]}')

V8 = ('{"matcher_version":"1.0.0","lexicon_version":"' + LV + '","norm_version":"NORM-1","tok_version":"TOK-1",'
      '"text":"","tokens":[],"spans":[],"unmatched_token_indices":[],"flags":[]}')


def alias_row(aid, node, surface, atype, weight, cmode, fuzzy):
    toks = tokenize(surface)
    norms = [norm_token(t["surface"]) for t in toks]
    return (aid, node, surface, US.join(norms), len(toks), atype, weight, cmode,
            fuzzy, "pub", None, "active")


def make_db(path, rows, nodes):
    conn = sqlite3.connect(path)
    ensure_schema(conn)
    conn.executemany("INSERT INTO conflict_node(node_id,label,kind) VALUES(?,?,'concept')",
                     [(n, n) for n in nodes])
    conn.execute(ALIAS_DDL)
    conn.executemany("INSERT INTO alias VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def run_vector(m, name, text, expected):
    got = m.match_json(text).decode("utf-8")
    want = expected.replace(LV, m.lexicon_version)
    if got != want:
        print(f"FAIL {name}")
        for i, (a, b) in enumerate(zip(got, want)):
            if a != b:
                print(f"  first divergence at byte {i}: got {got[i:i+70]!r} want {want[i:i+70]!r}")
                break
        else:
            print(f"  length differs: got {len(got)} want {len(want)}")
        sys.exit(1)
    print(f"PASS {name}  ({len(got)} bytes, byte-exact)")


def main():
    # §4 normative tokenizer consequences
    assert [t["surface"] for t in tokenize("Post-op")] == ["Post", "op"]
    assert [t["surface"] for t in tokenize("0.5")] == ["0.5"]
    assert [t["surface"] for t in tokenize("1,000")] == ["1,000"]
    assert [t["surface"] for t in tokenize(".5")] == ["5"]
    assert [t["surface"] for t in tokenize("N2O/O2")] == ["N2O", "O2"]
    assert [t["surface"] for t in tokenize("Parkinson's")] == ["Parkinson's"]
    assert norm_token("Parkinson's") == "parkinson" and norm_token("'s") == ""
    print("PASS §4/§3 tokenizer + normalizer consequences")

    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "lex.db")
        art = os.path.join(td, "artifacts")
        nodes = sorted({r[1] for r in LEXICON})
        make_db(db, [alias_row(*r) for r in LEXICON], nodes)

        rep = compile_lexicon(db, art)
        assert rep["ok"] and not rep["findings"], rep["findings"]
        assert re.fullmatch(r"[0-9a-f]{64}", rep["lexicon_version"])
        print(f"PASS compile (9 aliases, vocab {rep['counts']['vocab_size']}, no findings)")

        m = Matcher.load(art)
        assert m.lexicon_version == rep["lexicon_version"]

        run_vector(m, "V1 multi-token + hyphen", "Post-op nausea after appendicectomy.", V1)
        run_vector(m, "V2 ambiguous abbrev    ", "Query PE.", V2)
        run_vector(m, "V3 caps case fails     ", "ruled out pe", V3)
        run_vector(m, "V4 fuzzy corrected     ", "urgent apendicectomy", V4)
        run_vector(m, "V5 fuzzy suppressed    ", "started hydralazne", V5)
        run_vector(m, "V6 leftmost-longest    ", "acute appendicitis confirmed", V6)
        run_vector(m, "V7 possessive          ", "Parkinson's disease", V7)
        run_vector(m, "V8 empty input         ", "", V8)

        # §8.2 determinism: repeat + fresh load byte-identical
        a = m.match_json("Post-op nausea after appendicectomy.")
        b = Matcher.load(art).match_json("Post-op nausea after appendicectomy.")
        assert a == b
        print("PASS determinism (repeat match + fresh load byte-identical)")

        # §9 matcher error contract
        for bad, code in [(123, "E_INVALID_INPUT"), ("x\ud800y", "E_INVALID_INPUT"),
                          ("a" * 4097, "E_INPUT_TOO_LONG")]:
            try:
                m.match(bad)
                print(f"FAIL: expected {code}"); sys.exit(1)
            except LexError as e:
                assert e.code == code, f"got {e.code}, want {code}"
        try:
            Matcher.load(os.path.join(td, "nowhere"))
            print("FAIL: expected E_ARTIFACT_MISSING"); sys.exit(1)
        except LexError as e:
            assert e.code == "E_ARTIFACT_MISSING"
        meta_path = os.path.join(art, "lexicon.meta.json")
        meta = json.loads(open(meta_path, encoding="utf-8").read())
        meta["norm_version"] = "NORM-0"
        open(meta_path, "w", encoding="utf-8").write(json.dumps(meta))
        try:
            Matcher.load(art)
            print("FAIL: expected E_ARTIFACT_MISMATCH"); sys.exit(1)
        except LexError as e:
            assert e.code == "E_ARTIFACT_MISMATCH"
        print("PASS §9 error contract (invalid input / too long / artifact missing+mismatch)")

        # §6.3 compiler validation: collect ALL findings; fail ⇒ report only, no artifacts
        db2 = os.path.join(td, "bad.db")
        art2 = os.path.join(td, "artifacts2")
        bad_nodes = ["pub:x1"] + [f"pub:a{i}" for i in range(1, 7)]
        bad_rows = [
            alias_row(201, "pub:x1", "pe", "abbrev", 0.8, "fold", 1),          # C1
            alias_row(202, "pub:x1", "the", "synonym", 0.9, "fold", 1),        # C2
            alias_row(203, "pub:ghost", "ghostterm", "preferred", 1.0, "fold", 1),  # C3
            (204, "pub:x1", "appendix", "wrongseq", 1, "preferred", 1.0, "fold", 1,
             "pub", None, "active"),                                           # C4
            alias_row(205, "pub:x1", "'s", "variant", 0.5, "fold", 1),         # EMPTY_NORM (+C1)
        ] + [alias_row(210 + i, f"pub:a{i}", "amb", "synonym", 0.9, "fold", 1)
             for i in range(1, 6)]                                             # AMBIGUOUS_NORM
        bad_rows.append((216, "pub:a6", "amb", "amb", 1, "synonym", 0.9, "fold", 1,
                         "user", None, "active"))                              # ORIGIN_SHADOW
        make_db(db2, bad_rows, bad_nodes)
        rep2 = compile_lexicon(db2, art2)
        codes = {(f["level"], f["code"]) for f in rep2["findings"]}
        for want in [("ERROR", "C1"), ("ERROR", "C2"), ("ERROR", "C3"), ("ERROR", "C4"),
                     ("ERROR", "EMPTY_NORM"), ("WARN", "AMBIGUOUS_NORM"), ("WARN", "ORIGIN_SHADOW")]:
            assert want in codes, f"missing finding {want}: {sorted(codes)}"
        assert not rep2["ok"]
        assert os.path.exists(os.path.join(art2, "compile_report.json"))
        assert not os.path.exists(os.path.join(art2, "lexicon.json")), \
            "artifacts must not be written when ERRORs exist"
        print("PASS §6.3 compiler validation (C1–C4 + EMPTY_NORM errors, warns, report-only on fail)")

        print("\nALL ACCEPTANCE TESTS PASS — VINKONA-LEX-01 §11 conformant")


if __name__ == "__main__":
    main()
