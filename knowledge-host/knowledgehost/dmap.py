"""VINKONA-DMAP-01 — drug-source → conflict / mechanism / indication import mapping.

Offline, write-time.  Consumes normalized RxClass / MED-RT / FDASPL relation rows and the
ONC High-Priority DDI list; emits VINKONA nodes and edges consumed by VINKONA-CONF-01 (the
conflict checker) and the retrieval graph.

**This is the ONE drug-aware module.**  Everything it touches downstream is domain-neutral
(the CONF-01 ``administers`` scope link, the proposed→ratified gate).  Medications get a
dedicated importer because they are a zero-fault-tolerance safety surface — but the safety
comes from structure, not from trusting this code: per R1, every firing edge it mints enters
as ``status='proposed'``, which CONF-01 §5.1 guarantees can only ``flag_for_human``, never
fire or clear, until a human ratifies it.  No machine-minted safety edge is ever authoritative
on arrival.

The central rule (§1) is routing each RxClass relation family to the RIGHT place:
  * class/grouper (ATC, MeSH-PA, VA, has_EPC) → ``member_of`` hierarchy — scaffold, never fires;
  * mechanism (has_MoA, has_PE)               → mechanism node + ``acts_via`` — substrate, never fires;
  * pharmacokinetics (has_PK)                 → pathway node + ``has_pk`` — future PK substrate;
  * MED-RT ``CI_with``                        → a ``conflict_edge`` (contraindicated, drug×condition) — FIRES once ratified;
  * ``may_treat`` / ``may_prevent``           → indication edges (upstream of the action) — not conflicts;
  * ONC-High DDI pair                         → a SYMMETRIC pair of ``conflict_edge`` (drug×drug) — FIRES once ratified.

If the importer treated ``has_MoA`` as a conflict the checker would fire on mechanism
membership; if it treated ``CI_with`` as a class, contraindications would never fire.  Keeping
them in their lanes is the whole point.  Unresolvable drug/condition strings (R5) become
VINKONA-CONF-02 ``unresolved-mapping`` research items — a dropped contraindication is invisible
harm, so nothing is ever silently discarded.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from . import conflict

# ── firing families default to these until a ratifying reviewer sets the authoritative one (R4) ──
_CI_DEFAULT_SEVERITY = "caution"        # MED-RT CI_with carries no severity
_ONC_DEFAULT_SEVERITY = "severe"        # ONC-High is a high-priority list by definition

# small MoA/PE → conditionality_class lookup (§4); unknown ⇒ 'none'
_CONDITIONALITY = {
    "acute_competition": ("agonism", "agonist", "antagonism", "antagonist", "blocker",
                          "receptor", "competition"),
    "steady_state": ("inhibition", "inhibitor", "induction", "inducer", "enzyme"),
}

_INDICATION = {"may_treat": "indicated_for", "may_prevent": "prevents"}
_GROUPER_RELA = {"ATC": "ATC", "MeSH_PA": "MeSH-PA", "has_VA": "VA", "has_EPC": "EPC"}


class DmapError(Exception):
    def __init__(self, code: str, message: str):
        self.code, self.message = code, message
        super().__init__(f"{code}: {message}")


EXTRA_SCHEMA = """
CREATE TABLE IF NOT EXISTS indication (
  action     TEXT NOT NULL,
  condition  TEXT NOT NULL,
  relation   TEXT NOT NULL CHECK (relation IN ('indicated_for','prevents')),
  source_ref TEXT,
  PRIMARY KEY (action, condition, relation)
);
CREATE TABLE IF NOT EXISTS has_pk (
  substance  TEXT NOT NULL,
  pathway    TEXT NOT NULL,
  source_ref TEXT,
  PRIMARY KEY (substance, pathway)
);
CREATE TABLE IF NOT EXISTS node_xref (
  node_id TEXT NOT NULL,
  xref    TEXT NOT NULL,
  PRIMARY KEY (node_id, xref)
);
CREATE TABLE IF NOT EXISTS dmap_research (   -- staged CONF-02 'unresolved-mapping' items (R5)
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  kind       TEXT NOT NULL,
  subject    TEXT,
  rela       TEXT,
  source     TEXT,
  target_str TEXT,
  failed_side TEXT NOT NULL,
  source_ref TEXT
);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    """CONF-01 tables (shared) + this importer's own output tables."""
    conflict.ensure_schema(conn)
    conn.executescript(EXTRA_SCHEMA)
    conn.commit()


def _local(node_id: str) -> str:
    """Drop the authority/kind prefix for building a compact edge_id."""
    return node_id.split(":", 1)[1] if ":" in node_id else node_id


def conditionality_of(label: str) -> str:
    low = (label or "").lower()
    for cls, needles in _CONDITIONALITY.items():
        if any(n in low for n in needles):
            return cls
    return "none"


class _Result:
    def __init__(self):
        self.nodes: dict = {}          # node_id -> {id, kind, label, xref, conditionality_class?}
        self.mechanisms: dict = {}     # mechanism_id -> row for the `mechanism` table
        self.member_of: list = []
        self.acts_via: list = []
        self.has_pk: list = []
        self.conflict_edges: dict = {} # edge_id -> row (dedup by id)
        self.indications: list = []
        self.research: list = []

    def add_node(self, node_id, kind, label="", xref=None, **extra):
        n = self.nodes.setdefault(node_id, {"id": node_id, "kind": kind, "label": label,
                                            "xref": []})
        if label and not n["label"]:
            n["label"] = label
        for x in (xref or []):
            if x not in n["xref"]:
                n["xref"].append(x)
        n.update({k: v for k, v in extra.items() if v is not None})
        return node_id

    def as_dict(self) -> dict:
        for n in self.nodes.values():
            n["xref"].sort()
        return {
            "nodes": [self.nodes[k] for k in sorted(self.nodes)],
            "mechanisms": [self.mechanisms[k] for k in sorted(self.mechanisms)],
            "edges": (sorted(self.member_of, key=lambda e: (e["from"], e["to"]))
                      + sorted(self.acts_via, key=lambda e: (e["from"], e["to"], e["role"]))
                      + sorted(self.has_pk, key=lambda e: (e["from"], e["to"]))
                      + [self.conflict_edges[k] for k in sorted(self.conflict_edges)]
                      + sorted(self.indications, key=lambda e: (e["from"], e["to"], e["kind"]))),
            "conflict_edges": [self.conflict_edges[k] for k in sorted(self.conflict_edges)],
            "research_items": self.research,
        }


def _ci_edge(subject, condition, source_ref, subst_label, cond_label):
    return {"kind": "conflict_edge",
            "edge_id": "cirt:%s:%s" % (_local(subject), _local(condition)),
            "subject": subject, "relation_type": "contraindicated",
            "severity": _CI_DEFAULT_SEVERITY, "status": "proposed", "mechanism_id": None,
            "fire_when": {"op": "presence", "pred": condition},
            "authority": "pub", "source_ref": source_ref,
            "rationale": "MED-RT: %s contraindicated with %s" % (subst_label, cond_label)}


def map_substance(record: dict, resolver) -> dict:
    """Map one source substance record to VINKONA nodes/edges (§2 routing).  ``record`` =
    ``{"name","rxcui"?,"atc":[...],"version","relations":[{"rela","source","target"}...]}``.
    ``resolver(kind, name) -> {"node_id",...} | None`` maps a source string to a canon node
    (R6 dedup + authority-prefix crosswalk live inside the resolver); None ⇒ research item."""
    r = _Result()
    ver = record.get("version", "")
    sub = resolver("substance", record["name"])
    if not sub:
        r.research.append({"type": "unresolved-mapping", "kind": "substance",
                           "subject": None, "rela": None, "source": None,
                           "target_str": record["name"], "failed_side": "subject",
                           "source_ref": None})
        return r.as_dict()                      # can't attach anything without a subject
    subject = sub["node_id"]
    subst_label = sub.get("label", record["name"])
    xref = [f"RXCUI:{record['rxcui']}"] if record.get("rxcui") else []
    xref += [f"ATC:{a}" for a in (record.get("atc") or [])]
    r.add_node(subject, "substance", subst_label, xref)

    def research(kind, rela, source, target, side):
        r.research.append({"type": "unresolved-mapping", "kind": kind, "subject": subject,
                           "rela": rela, "source": source, "target_str": target,
                           "failed_side": side, "source_ref": f"{source}/{rela}/{ver}"})

    for rel in record.get("relations", []):
        rela, source, target = rel["rela"], rel.get("source", "MED-RT"), rel["target"]
        sref = f"{source}/{rela}/{ver}"

        if rela in _GROUPER_RELA or rela == "ATC":          # class/grouper → member_of (scaffold)
            gtype = _GROUPER_RELA.get(rela, "ATC")
            g = resolver("class", target)
            if not g:
                research("class", rela, source, target, "target"); continue
            r.add_node(g["node_id"], "class", g.get("label", target))
            r.member_of.append({"kind": "member_of", "from": subject, "to": g["node_id"],
                                "grouper_type": gtype, "source_ref": sref})

        elif rela in ("has_MoA", "has_PE"):                 # mechanism → node + acts_via (substrate)
            role = "moa" if rela == "has_MoA" else "pe"
            m = resolver("mechanism", target)
            if not m:
                research("mechanism", rela, source, target, "target"); continue
            mid = m["node_id"]
            cclass = m.get("conditionality_class") or conditionality_of(m.get("label", target))
            r.add_node(mid, "mechanism", m.get("label", target),
                       conditionality_class=cclass)
            r.mechanisms[mid] = {"mechanism_id": mid, "label": m.get("label", target),
                                 "explanation": m.get("explanation", ""),
                                 "conditionality_class": cclass}
            r.acts_via.append({"kind": "acts_via", "from": subject, "to": mid,
                               "role": role, "source_ref": sref})

        elif rela == "has_PK":                              # PK → pathway + has_pk (future substrate)
            p = resolver("pathway", target)
            if not p:
                research("pathway", rela, source, target, "target"); continue
            r.add_node(p["node_id"], "pathway", p.get("label", target))
            r.has_pk.append({"kind": "has_pk", "from": subject, "to": p["node_id"],
                             "source_ref": sref})

        elif rela == "CI_with":                             # THE conflict family (drug×condition)
            cond = resolver("condition", target)
            if not cond:
                research("condition", rela, source, target, "target"); continue
            cid = cond["node_id"]
            r.add_node(cid, "condition", cond.get("label", target))
            edge = _ci_edge(subject, cid, sref, subst_label, cond.get("label", target))
            r.conflict_edges[edge["edge_id"]] = edge

        elif rela in _INDICATION:                           # indication (upstream, not a conflict)
            cond = resolver("condition", target)
            if not cond:
                research("condition", rela, source, target, "target"); continue
            cid = cond["node_id"]
            r.add_node(cid, "condition", cond.get("label", target))
            action = sub.get("action") or "act:administer_" + _local(subject)
            r.add_node(action, "action", f"administer {subst_label}")  # action node must exist in graph
            r.indications.append({"kind": _INDICATION[rela], "from": action, "to": cid,
                                  "source_ref": sref})
        # unknown relas are ignored (import only what the §2 table maps)

    return r.as_dict()


def map_ddi_pair(a: dict, b: dict, *, version="2024",
                 relation_type="contraindicated") -> list:
    """ONC High-Priority ingredient×ingredient pair → a SYMMETRIC pair of proposed
    conflict_edges (§5), so it fires regardless of which drug the card names.  Both
    default to `severe` (R4); use `antagonizes` where the interaction is efficacy-opposition."""
    if relation_type not in ("contraindicated", "antagonizes"):
        raise DmapError("E_BAD_RELATION", f"DDI relation_type must be contraindicated/antagonizes")
    out = []
    for x, y in ((a, b), (b, a)):
        out.append({"kind": "conflict_edge", "edge_id": "onc:%s:%s" % (_local(x["node_id"]),
                                                                       _local(y["node_id"])),
                    "subject": x["node_id"], "relation_type": relation_type,
                    "severity": _ONC_DEFAULT_SEVERITY, "status": "proposed", "mechanism_id": None,
                    "fire_when": {"op": "presence", "pred": y["node_id"]},
                    "authority": "pub", "source_ref": f"ONC-HPDDI/{version}",
                    "rationale": "ONC high-priority interaction: %s + %s"
                                 % (a.get("label", _local(a["node_id"])),
                                    b.get("label", _local(b["node_id"])))})
    return out


def write(conn: sqlite3.Connection, result: dict) -> dict:
    """Persist an import result into the CONF-01 + importer tables, idempotently (R6): every
    write is INSERT OR IGNORE / keyed so a same-source re-import is a no-op.  conflict_edges
    are validated by CONF-01's own grammar on the next Checker.load()."""
    import json as _json
    ensure_schema(conn)
    for n in result["nodes"]:
        conn.execute("INSERT OR IGNORE INTO conflict_node(node_id,label,kind) VALUES(?,?,?)",
                     (n["id"], n.get("label", ""), n["kind"]))
        for x in n.get("xref", []):
            conn.execute("INSERT OR IGNORE INTO node_xref(node_id,xref) VALUES(?,?)", (n["id"], x))
    for m in result.get("mechanisms", []):
        conn.execute("INSERT OR IGNORE INTO mechanism(mechanism_id,label,explanation,"
                     "conditionality_class) VALUES(?,?,?,?)",
                     (m["mechanism_id"], m["label"], m["explanation"], m["conditionality_class"]))
    counts = {"conflict_edges": 0, "member_of": 0, "acts_via": 0, "has_pk": 0,
              "indications": 0, "research": 0}
    for e in result["edges"]:
        k = e["kind"]
        if k == "member_of":
            conn.execute("INSERT OR IGNORE INTO member_of(child,grouper,grouper_type) VALUES(?,?,?)",
                         (e["from"], e["to"], e["grouper_type"])); counts["member_of"] += 1
        elif k == "acts_via":
            conn.execute("INSERT OR IGNORE INTO acts_via(substance,mechanism,role) VALUES(?,?,?)",
                         (e["from"], e["to"], e["role"])); counts["acts_via"] += 1
        elif k == "has_pk":
            conn.execute("INSERT OR IGNORE INTO has_pk(substance,pathway,source_ref) VALUES(?,?,?)",
                         (e["from"], e["to"], e.get("source_ref"))); counts["has_pk"] += 1
        elif k in ("indicated_for", "prevents"):
            conn.execute("INSERT OR IGNORE INTO indication(action,condition,relation,source_ref) "
                         "VALUES(?,?,?,?)", (e["from"], e["to"], k, e.get("source_ref")))
            counts["indications"] += 1
        elif k == "conflict_edge":
            conn.execute(
                "INSERT OR IGNORE INTO conflict_edge(edge_id,subject,relation_type,severity,"
                "fire_when,mechanism_id,status,authority,rationale,source_ref) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (e["edge_id"], e["subject"], e["relation_type"], e["severity"],
                 _json.dumps(e["fire_when"], separators=(",", ":")), e["mechanism_id"],
                 e["status"], e["authority"], e["rationale"], e["source_ref"]))
            counts["conflict_edges"] += 1
    for it in result.get("research_items", []):
        conn.execute("INSERT INTO dmap_research(kind,subject,rela,source,target_str,failed_side,"
                     "source_ref) VALUES(?,?,?,?,?,?,?)",
                     (it["kind"], it["subject"], it["rela"], it["source"], it["target_str"],
                      it["failed_side"], it["source_ref"]))
        counts["research"] += 1
    conn.commit()
    return counts
