"""Optional HNSW ANN index over the KB node embeddings (usearch).

Brute-force cosine streams the entire N×dim matrix per query — memory-bandwidth-bound,
~90ms at 1M×768 and growing linearly.  This swaps it for an HNSW graph: ~log(N) hops,
single-digit ms, flat as the corpus grows, and memory-mapped on load so the server no
longer keeps a multi-GB matrix resident.

Accuracy: embeddings are L2-normalised at write time, so cosine == inner product.  The
index uses the 'cos' metric and returns EXACT similarities (``1 - distance``) for the
candidates it surfaces — the only approximation is *recall* (which candidates are found),
governed by ``expansion_search``.  Callers over-fetch and keep the true top-k, so scores
are exact; recall is kept high by generous expansion.

OPTIONAL by design.  If usearch is not installed or no index file exists, the KB falls
back to the exact brute-force matmul, so this is a pure speed-up with identical results
(modulo HNSW recall).  Used on the READ paths only (search, reconcile); the distillation
write path keeps the exact in-RAM matrix, because a node added mid-run is not yet in the
persisted index and a missed identity match silently forks a duplicate node.
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger("knowledgehost.ann")

try:
    import numpy as np
    from usearch.index import Index
    _OK = True
except Exception:                                  # usearch absent -> brute-force fallback
    _OK = False


def available() -> bool:
    return _OK


class AnnIndex:
    """Maps usearch integer keys (row numbers) back to node ids."""

    def __init__(self, ids, index):
        self.ids = ids                             # row -> node_id
        self.index = index

    @classmethod
    def build(cls, ids, vecs, *, connectivity=32, expansion_add=128, expansion_search=128,
              dtype="f16"):
        # dtype is the STORED precision (f16 ≈ half the RAM of f32 with negligible cosine
        # error — the right default on a CPU box where index residency drives latency).
        idx = Index(ndim=int(vecs.shape[1]), metric="cos", dtype=dtype,
                    connectivity=connectivity, expansion_add=expansion_add,
                    expansion_search=expansion_search)
        idx.add(np.arange(len(ids), dtype=np.uint64),
                np.ascontiguousarray(vecs, dtype="f4"))   # added as f32, stored as dtype
        return cls(list(ids), idx)

    def query(self, q, k):
        """Top-k as [(node_id, cosine_sim)], highest first."""
        k = min(int(k), len(self.ids))
        if k <= 0:
            return []
        m = self.index.search(np.asarray(q, dtype="f4"), k)
        return [(self.ids[int(key)], 1.0 - float(d))
                for key, d in zip(m.keys, m.distances)]

    def save(self, path):
        self.index.save(path + ".usearch")
        with open(path + ".ids.json", "w") as f:
            json.dump(self.ids, f)

    @classmethod
    def load(cls, path, *, expansion_search=128, view=False):
        with open(path + ".ids.json") as f:
            ids = json.load(f)
        # view=False loads the whole index RESIDENT in RAM — every query is then RAM-speed
        # regardless of which graph regions it touches.  view=True memory-maps it (lower
        # RSS) but makes each *new* query page its traversal in from disk: catastrophic on
        # a RAM-tight CPU box (seconds/query).  Resident is the right default for a server.
        idx = Index.restore(path + ".usearch", view=view)
        try:
            idx.expansion_search = int(expansion_search)
        except Exception:
            pass
        return cls(ids, idx)

    def __len__(self):
        return len(self.ids)


def index_exists(path) -> bool:
    return os.path.exists(path + ".usearch") and os.path.exists(path + ".ids.json")


def build_from_kb(kb, path, *, connectivity=32, expansion_add=128, expansion_search=128,
                  dtype="f16", min_nodes=0, log_every=200_000) -> dict:
    """Stream every active, embedded node out of SQLite and build+save the index.

    Reads through the raw connection (under the KB lock) so a 1M-row scan streams instead
    of materialising in the locking proxy.  Returns a stats dict."""
    if not _OK:
        raise RuntimeError("usearch not installed — `pip install usearch` to build the "
                           "ANN index (the KB still works, just on brute-force search).")
    import time
    t0 = time.time()
    with kb._lock:
        n = kb._raw.execute("SELECT COUNT(*) FROM nodes WHERE status='active' "
                            "AND embedding IS NOT NULL").fetchone()[0]
        if n < min_nodes:
            return {"nodes": n, "built": False,
                    "reason": f"below ann_min_nodes ({min_nodes}); brute force is exact here"}
        row = kb._raw.execute("SELECT embedding FROM nodes WHERE status='active' "
                             "AND embedding IS NOT NULL LIMIT 1").fetchone()
        if not row:
            return {"nodes": 0, "built": False, "reason": "no embedded nodes"}
        dim = len(row["embedding"]) // 4
        vecs = np.empty((n, dim), dtype="f4")
        ids = []
        i = 0
        for r in kb._raw.execute("SELECT id, embedding FROM nodes WHERE status='active' "
                                 "AND embedding IS NOT NULL"):
            buf = r["embedding"]
            if len(buf) // 4 != dim:               # defensive: skip a malformed vector
                continue
            vecs[i] = np.frombuffer(buf, dtype="<f4")
            ids.append(r["id"])
            i += 1
            if i % log_every == 0:
                log.info("build-ann: read %d/%d node vectors", i, n)
    vecs = vecs[:i]
    log.info("build-ann: building HNSW over %d×%d (%s, connectivity=%d, expansion_add=%d)…",
             i, dim, dtype, connectivity, expansion_add)
    ann = AnnIndex.build(ids, vecs, connectivity=connectivity, expansion_add=expansion_add,
                         expansion_search=expansion_search, dtype=dtype)
    ann.save(path)
    return {"nodes": i, "dim": dim, "dtype": dtype, "built": True,
            "path": path + ".usearch", "elapsed_s": round(time.time() - t0, 1)}
