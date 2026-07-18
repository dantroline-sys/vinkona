"""Dedup sweep — repeat observations accumulate instead of duplicating.

Fifty documents describing the same procedure should not mint fifty cards.
The sweep groups ACTIVE cards by (node, card type, provenance-bundle set,
normalized content hash); within a group the oldest card survives, absorbs
the union of support and the sum of observations into observed_count /
last_observed (the reserved learned fields every card row carries), and the
rest flip to status='merged' — invisible to every status='active' read path,
reversible by flipping back.

Conservative by design: normalization is whitespace/case folding over the
title + canonical payload JSON, never fuzzy matching; cards whose support
spans different bundle sets never merge (a merge must not smuggle content
across bundle boundaries).

Run `rebuild-fts` on the host after a sweep that merged anything — cards_fts
does not track status flips.
"""
import hashlib
import json
import time


def _fold(v):
    if isinstance(v, str):
        return " ".join(v.split()).lower()
    if isinstance(v, list):
        return [_fold(x) for x in v]
    if isinstance(v, dict):
        return {k: _fold(v[k]) for k in sorted(v)}
    return v


def _norm_hash(row):
    payload = {}
    for col in ("criteria", "goal", "steps", "preconditions", "safety"):
        raw = row[col] if col in row.keys() else None
        if not raw:
            continue
        try:
            payload[col] = _fold(json.loads(raw))
        except (ValueError, TypeError):
            payload[col] = _fold(str(raw))
    basis = json.dumps({"title": _fold(row["title"] or ""), "payload": payload},
                       sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(basis.encode()).hexdigest()


def _support_docs(raw):
    try:
        return [e.get("doc_id") for e in json.loads(raw or "[]")
                if isinstance(e, dict) and e.get("doc_id")]
    except (ValueError, TypeError):
        return []


def sweep(kb, bundle=None, dry=False, log_fn=None):
    """Merge duplicate active cards.  `bundle` restricts to cards whose support
    lives entirely in that bundle.  Returns stats."""
    say = log_fn or (lambda m: None)
    doc_bundle = {r["doc_id"]: (r["bundle"] or "base") for r in kb.db.execute(
        "SELECT doc_id, bundle FROM source_registry")}
    rows = kb.db.execute("SELECT * FROM procedure_cards WHERE status='active' "
                         "ORDER BY created_at, id").fetchall()
    groups = {}
    for r in rows:
        docs = _support_docs(r["support"])
        bundles = frozenset(doc_bundle.get(d, "base") for d in docs) or \
            frozenset({"base"})
        if bundle is not None and bundles != frozenset({bundle}):
            continue
        ctype = (r["card_type"] if "card_type" in r.keys() else None) or "procedure"
        key = (r["node_id"], ctype, bundles, _norm_hash(r))
        groups.setdefault(key, []).append(r)

    stats = {"cards": len(rows), "groups": 0, "merged": 0}
    now = time.time()
    for key, group in groups.items():
        if len(group) < 2:
            continue
        stats["groups"] += 1
        survivor, dups = group[0], group[1:]
        seen, support = set(), []
        for r in group:
            try:
                entries = json.loads(r["support"] or "[]")
            except (ValueError, TypeError):
                entries = []
            for e in entries:
                d = e.get("doc_id") if isinstance(e, dict) else None
                if d and d not in seen:
                    seen.add(d)
                    support.append(e)
        observed = sum(max(int(r["observed_count"] or 0), 1) for r in group)
        last = max(float(r["updated_at"] or 0) for r in group)
        say(f"merge {len(group)}→1 on {survivor['node_id']}: "
            f"{(survivor['title'] or '')[:60]!r} (observed={observed})")
        stats["merged"] += len(dups)
        if dry:
            continue
        kb.db.execute(
            "UPDATE procedure_cards SET support=?, observed_count=?, "
            "last_observed=?, updated_at=? WHERE id=?",
            (json.dumps(support, ensure_ascii=False), observed, last, now,
             survivor["id"]))
        kb.db.executemany(
            "UPDATE procedure_cards SET status='merged', updated_at=? WHERE id=?",
            [(now, r["id"]) for r in dups])
    if not dry:
        kb.db.commit()
    return stats
