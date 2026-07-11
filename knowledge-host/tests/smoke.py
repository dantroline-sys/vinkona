"""End-to-end self-test — zero pip installs required.

Proves the Phase-1 path from KNOWLEDGE.md: ingest a small document set ->
hybrid retrieval -> cited passages, on the sqlite backend.  Runs twice:

  1. sparse-only (the embed endpoint is "down") — the stdlib guarantee;
  2. dense+sparse with a deterministic FAKE embedder — exercises the vector
     arm, RRF fusion and the reranker without needing a live nomic server.

    python3 tests/smoke.py [/tmp/kb-fixtures]
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledgehost.config import load_config
from knowledgehost.embed import Embedder
from knowledgehost import ingest as ingest_mod
from knowledgehost.store import make_store
from knowledgehost.tools import Tools


class FakeEmbedder:
    """Deterministic hashing embedder — no server. Bag-of-terms hashed into a
    fixed-dim L2-normalized vector, so related text lands nearby in cosine."""
    DIM = 256

    def _vec(self, text):
        import math, re
        v = [0.0] * self.DIM
        for t in re.split(r"\W+", text.lower()):
            if len(t) > 2:
                v[hash(t) % self.DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n else None

    def embed_one(self, text, task="document"):
        return self._vec(text)

    def embed_many(self, texts, task="document"):
        return [self._vec(t) for t in texts]


def run(fixtures, embedder, label):
    tmp = tempfile.mkdtemp(prefix="kb-smoke-")
    cfg = load_config(None)
    cfg["db_path"] = os.path.join(tmp, "index.db")
    cfg["sources"] = [fixtures]
    cfg["backend"] = "sqlite"
    cfg["min_confidence"] = 0.0

    store = make_store(cfg)
    stats = ingest_mod.crawl(store, embedder, cfg)
    assert stats["chunks"] > 0, "no chunks ingested"
    tools = Tools(store, embedder, cfg)

    def search(q, **kw):
        r = tools.call("kb_search", {"query": q, **kw})
        assert r["ok"], r
        return json.loads(r["result"])

    # 1. a Wikipedia-style lookup hits the right doc, cited
    r = search("who discovered the Krebs cycle", k=3)
    assert r["passages"], "no passages for Krebs query"
    top = r["passages"][0]
    assert "krebs" in (top["text"] + top["title"]).lower(), top
    assert top["path_or_url"], "passage not cited"
    expect_dense = embedder.embed_one("probe", "query") is not None
    assert r["dense_used"] == expect_dense, (r["dense_used"], expect_dense)

    # 2. exact-term recall (FTS strength): a proper noun
    r = search("Paula Denise Agnus chipset")
    assert any("agnus" in p["text"].lower() for p in r["passages"]), r

    # 3. a miss trips the low-confidence / web-fallback gate
    cfg["min_confidence"] = 0.99
    r = search("quarterly revenue of a fictional corporation xyzzy")
    assert r["low_confidence"] and "note" in r, r

    store.close()
    print(f"  [{label}] ok — {stats['docs']} docs / {stats['chunks']} chunks; "
          f"top score {top['score']}")


def main():
    fixtures = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kb-fixtures"
    if not os.path.isdir(fixtures):
        sys.exit(f"fixtures not found: {fixtures}\n"
                 f"run: bash tests/make_fixtures.sh {fixtures}")
    print("knowledge-host smoke test")
    run(fixtures, Embedder({**load_config(None), "embed_url": "http://127.0.0.1:1"}),
        "sparse-only")
    run(fixtures, FakeEmbedder(), "dense+sparse")
    print("ALL PASS")


if __name__ == "__main__":
    main()
