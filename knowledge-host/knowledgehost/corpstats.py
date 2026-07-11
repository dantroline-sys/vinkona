"""VINKONA-STAT-01 — corpus statistics pass (card salience & concept co-occurrence).

An **offline global pass**: a pure function of a postings snapshot plus a fixed parameter
set, never in the query hot path.  Derives two weight sets and materializes them onto
graph edges via shadow tables + one atomic swap:

  1. card→concept **salience** — normalized sublinear-TF×IDF per card-concept reference
     (concept-coverage ranking / spreading-activation seed mass);
  2. concept→concept **co-occurrence** — PPMI strength on ``co_occurs_with`` edges
     (the Associative Fallback tier / associative band).

No ML, no GPU, no network, no learned weight anywhere: ``K_COOC`` (top-K salient concepts
per card), ``MIN_COOC`` (small-sample floor) and the PPMI clamp are the complete guard.
The §11 hub lesson is the point: a concept in 9 of 10 cards gets ZERO surviving edges —
raw co-occurrence would have crowned it the best-connected node; PMI+clamp discards every
hub association as no-better-than-chance.

This pass MUST NOT create/modify/delete ontological or causal edges — it writes only its
own two output tables.  The postings ledger is produced by VINKONA-ING-01 (not yet drafted);
until then G0 validates concept IDs against the **canon registry** (``conflict_node``,
VINKONA-CONF-01) by default — parameterizable via ``node_table``/``node_ids``.

Determinism (§8): fixed snapshot + params ⇒ byte-identical stats_report.json across runs,
platforms, thread counts.  All ordering is total (concept-id byte order + tie-breaks);
rounding is half-even to 6 dp; comparisons (floor, clamp) use unrounded values.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from pathlib import Path

ALGO_VERSION = "VINKONA-STAT-01/1.0"
_SEP = "\x1e"                       # U+001E record separator (§7 canon string)

DEFAULT_K_COOC = 20
DEFAULT_MIN_COOC = 3


class StatError(Exception):
    """§9 error contract: E_UNKNOWN_CONCEPT | E_BAD_PARAM | E_POSTINGS_MALFORMED |
    E_SWAP_FAILED.  N = 0 is NOT an error (empty result set is a valid build)."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _check_params(k_cooc, min_cooc) -> None:
    for name, val, lo in (("K_COOC", k_cooc, 2), ("MIN_COOC", min_cooc, 1)):
        if isinstance(val, bool) or not isinstance(val, int) or val < lo:
            raise StatError("E_BAD_PARAM", f"{name} must be an integer >= {lo}, got {val!r}")


def stats_version(postings, k_cooc: int, min_cooc: int) -> str:
    """§7: sha256 over algo_version ␞ params_json ␞ postings-body — any change to the
    postings content, either parameter, or the algorithm string changes the version."""
    rows = sorted(postings, key=lambda p: (p[0], p[1]))
    body = "\n".join(json.dumps([c, cid, int(n)], separators=(",", ":"), ensure_ascii=False)
                     for c, cid, n in rows)
    params_json = '{"K_COOC":%d,"MIN_COOC":%d}' % (k_cooc, min_cooc)
    canon = ALGO_VERSION + _SEP + params_json + _SEP + body
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def compute(postings, *, k_cooc: int = DEFAULT_K_COOC,
            min_cooc: int = DEFAULT_MIN_COOC) -> dict:
    """Stages G0(shape)–G5 as a pure function of ``[(card_id, concept_id, local_count)]``.
    Returns salience rows, surviving co-occurrence edges, and the two drop ledgers —
    every list already in its §6.3/§11.3 deterministic order."""
    _check_params(k_cooc, min_cooc)
    seen: set = set()
    by_card: dict = {}
    for card, concept, count in postings:
        if (card, concept) in seen:
            raise StatError("E_POSTINGS_MALFORMED", f"duplicate posting ({card}, {concept})")
        seen.add((card, concept))
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise StatError("E_POSTINGS_MALFORMED",
                            f"local_count must be an integer >= 1 for ({card}, {concept})")
        by_card.setdefault(card, []).append((concept, count))

    n_cards = len(by_card)
    out = {"N": n_cards, "salience": [], "cooccurrence": [],
           "floor_drops": [], "clamp_drops": []}
    if n_cards == 0:                                   # §G0: valid empty result, not an error
        return out

    # G1 — document frequency & smoothed IDF (always > 0, so products stay well-defined)
    df: dict = {}
    for rows in by_card.values():
        for concept, _ in rows:
            df[concept] = df.get(concept, 0) + 1
    idf = {c: math.log((n_cards + 1) / (d + 1)) + 1.0 for c, d in df.items()}

    # G2 — raw & normalized salience;  G3 — salient set;  G4 — sdf/cooc accumulation.
    # The concept×concept matrix is never materialized: only observed pairs get entries,
    # so an unobserved pair is a true absence of evidence, not a stored zero.
    sdf: dict = {}
    cooc: dict = {}
    for card, rows in by_card.items():
        raw = [(concept, (1.0 + math.log(count)) * idf[concept]) for concept, count in rows]
        m = max(r for _, r in raw)                     # strictly positive per-card scale
        for concept, r in raw:
            out["salience"].append({"card_id": card, "concept_id": concept,
                                    "weight": round(r / m, 6)})
        # (raw desc, concept_id asc), truncate to K_COOC: hubs carry the lowest IDF, so
        # they fall out of S first in concept-dense cards — self-exclusion by ordering.
        salient = sorted([c for c, _ in sorted(raw, key=lambda t: (-t[1], t[0]))[:k_cooc]])
        for c in salient:
            sdf[c] = sdf.get(c, 0) + 1
        for i in range(len(salient)):
            for j in range(i + 1, len(salient)):
                pair = (salient[i], salient[j])        # a < b by construction
                cooc[pair] = cooc.get(pair, 0) + 1

    out["salience"].sort(key=lambda e: (e["card_id"], e["concept_id"]))

    # G5 — floor, PMI over SALIENT-set marginals (sdf, NOT df — the pair counts are over
    # salient sets, so df marginals would yield a malformed PMI), then the PPMI clamp.
    for (a, b), count in sorted(cooc.items()):
        if count < min_cooc:                           # small-sample floor: one sighting
            out["floor_drops"].append(                 # would otherwise score maximally
                {"concept_a": a, "concept_b": b, "cooc_count": count})
            continue
        pmi = math.log((count * n_cards) / (sdf[a] * sdf[b]))
        if pmi <= 0.0:                                 # at/below chance ⇒ no positive signal
            out["clamp_drops"].append(
                {"concept_a": a, "concept_b": b, "pmi": round(pmi, 6), "cooc_count": count})
            continue
        out["cooccurrence"].append(
            {"concept_a": a, "concept_b": b, "ppmi": round(pmi, 6), "cooc_count": count})
    return out


def render_report(version: str, k_cooc: int, min_cooc: int, result: dict) -> str:
    """§6.3 stats_report.json — the byte-exact conformance artifact.  Canonical JSON:
    compact separators, fixed key order, arrays pre-sorted, non-ASCII raw, numbers as the
    shortest round-tripping repr of the already-rounded value."""
    doc = {"stats_version": version, "algo_version": ALGO_VERSION,
           "params": {"K_COOC": k_cooc, "MIN_COOC": min_cooc},
           "corpus": {"N": result["N"]},
           "salience": result["salience"],
           "cooccurrence": result["cooccurrence"]}
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":"))


_SALIENCE_DDL = """CREATE TABLE {name} (
  card_id     TEXT NOT NULL,
  concept_id  TEXT NOT NULL,
  weight      REAL NOT NULL CHECK (weight > 0.0 AND weight <= 1.0),
  PRIMARY KEY (card_id, concept_id)
)"""
_COOC_DDL = """CREATE TABLE {name} (
  concept_a   TEXT NOT NULL,
  concept_b   TEXT NOT NULL,
  ppmi        REAL NOT NULL CHECK (ppmi > 0.0),
  cooc_count  INTEGER NOT NULL CHECK (cooc_count >= 1),
  PRIMARY KEY (concept_a, concept_b),
  CHECK (concept_a < concept_b)
)"""


def _materialize(conn: sqlite3.Connection, version: str, k_cooc: int, min_cooc: int,
                 result: dict) -> None:
    """§6.2: populate shadow tables, verify row counts, then swap shadow → live inside a
    single BEGIN IMMEDIATE…COMMIT.  On any error the prior live tables stay intact — a
    reader only ever sees one fully-consistent stats_version."""
    conn.execute("DROP TABLE IF EXISTS card_concept_salience_shadow")
    conn.execute("DROP TABLE IF EXISTS concept_cooccurrence_shadow")
    conn.execute(_SALIENCE_DDL.format(name="card_concept_salience_shadow"))
    conn.execute(_COOC_DDL.format(name="concept_cooccurrence_shadow"))
    conn.execute("CREATE TABLE IF NOT EXISTS stats_meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
    conn.executemany(
        "INSERT INTO card_concept_salience_shadow(card_id,concept_id,weight) VALUES(?,?,?)",
        [(e["card_id"], e["concept_id"], e["weight"]) for e in result["salience"]])
    conn.executemany(
        "INSERT INTO concept_cooccurrence_shadow(concept_a,concept_b,ppmi,cooc_count) "
        "VALUES(?,?,?,?)",
        [(e["concept_a"], e["concept_b"], e["ppmi"], e["cooc_count"])
         for e in result["cooccurrence"]])
    conn.commit()

    ns = conn.execute("SELECT COUNT(*) FROM card_concept_salience_shadow").fetchone()[0]
    nc = conn.execute("SELECT COUNT(*) FROM concept_cooccurrence_shadow").fetchone()[0]
    if ns != len(result["salience"]) or nc != len(result["cooccurrence"]):
        conn.execute("DROP TABLE IF EXISTS card_concept_salience_shadow")
        conn.execute("DROP TABLE IF EXISTS concept_cooccurrence_shadow")
        conn.commit()
        raise StatError("E_SWAP_FAILED",
                        f"shadow row-count mismatch (salience {ns}/{len(result['salience'])}, "
                        f"cooccurrence {nc}/{len(result['cooccurrence'])})")

    old_iso = conn.isolation_level
    conn.isolation_level = None                        # manual transaction control
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DROP TABLE IF EXISTS card_concept_salience")
        conn.execute("DROP TABLE IF EXISTS concept_cooccurrence")
        conn.execute("ALTER TABLE card_concept_salience_shadow RENAME TO card_concept_salience")
        conn.execute("ALTER TABLE concept_cooccurrence_shadow RENAME TO concept_cooccurrence")
        conn.execute("DELETE FROM stats_meta")
        conn.executemany("INSERT INTO stats_meta(k,v) VALUES(?,?)", [
            ("stats_version", version), ("algo_version", ALGO_VERSION),
            ("K_COOC", str(k_cooc)), ("MIN_COOC", str(min_cooc)),
            ("N", str(result["N"])),
            ("built_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))])
        conn.execute("COMMIT")
    except Exception as e:
        conn.execute("ROLLBACK")
        raise StatError("E_SWAP_FAILED", f"{type(e).__name__}: {e}")
    finally:
        conn.isolation_level = old_iso


def build(db, out_dir=None, *, k_cooc: int = DEFAULT_K_COOC,
          min_cooc: int = DEFAULT_MIN_COOC,
          node_table: str = "conflict_node", node_ids=None) -> dict:
    """``vinkona-stat build``: read the posting ledger, validate every concept against the
    node source (G0), compute STAT-1, atomically materialize, and (if ``out_dir``) write
    ``stats_report.json`` (normative, byte-exact) plus ``stats_drops.json`` (non-normative
    drop ledger for ops eyes).  ``db`` is a path or an open sqlite3 connection."""
    _check_params(k_cooc, min_cooc)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", node_table):
        raise StatError("E_BAD_PARAM", f"bad node_table name: {node_table!r}")
    own = isinstance(db, (str, Path))
    conn = sqlite3.connect(str(db)) if own else db
    try:
        postings = [(r[0], r[1], r[2]) for r in conn.execute(
            "SELECT card_id, concept_id, local_count FROM posting")]
        known = (set(node_ids) if node_ids is not None else
                 {r[0] for r in conn.execute(f"SELECT node_id FROM {node_table}")})
        unknown = sorted({cid for _, cid, _ in postings} - known)
        if unknown:
            raise StatError("E_UNKNOWN_CONCEPT",
                            "postings reference unknown concept(s): " + ", ".join(unknown))

        version = stats_version(postings, k_cooc, min_cooc)
        result = compute(postings, k_cooc=k_cooc, min_cooc=min_cooc)
        _materialize(conn, version, k_cooc, min_cooc, result)

        report = render_report(version, k_cooc, min_cooc, result)
        if out_dir:
            p = Path(out_dir)
            p.mkdir(parents=True, exist_ok=True)
            (p / "stats_report.json").write_text(report, encoding="utf-8")
            (p / "stats_drops.json").write_text(
                json.dumps({"floor": result["floor_drops"],
                            "ppmi_clamp": result["clamp_drops"]},
                           ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        return {"stats_version": version, "N": result["N"],
                "salience_rows": len(result["salience"]),
                "cooccurrence_edges": len(result["cooccurrence"]),
                "floor_drops": len(result["floor_drops"]),
                "clamp_drops": len(result["clamp_drops"]),
                "report": report,
                "drops": {"floor": result["floor_drops"],
                          "ppmi_clamp": result["clamp_drops"]}}
    finally:
        if own:
            conn.close()
