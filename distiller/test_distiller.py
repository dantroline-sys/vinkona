"""Distiller add-in, end to end against a scratch vinur store+kb: digest
caching, digest-first prompts (byte-stable prefix), one-card-factory writes,
checkpoint semantics, fail-open on a dead backend, dedup sweep accumulation.
Stdlib + a vinur checkout; stub LM, no services."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VINUR = Path(os.environ.get("VINUR_REPO", "/home/user/vinur"))
sys.path.insert(0, str(REPO))

from distiller import bootstrap  # noqa: E402

bootstrap(VINUR)

from distiller import digest, driver, merge  # noqa: E402
from knowledgehost import config as khconfig, distill as D  # noqa: E402
from knowledgehost.kb import KB  # noqa: E402
from knowledgehost.store import make_store  # noqa: E402

FAILED = []


def check(label, cond):
    print(("  ok  " if cond else "FAIL  ") + label)
    if not cond:
        FAILED.append(label)


class StubLM:
    max_tokens = 3072

    def __init__(self):
        self.extracts = []          # (system, user)
        self.glosses = []
        self.down = False

    def _content(self, system, user, schema, max_tokens):
        if self.down:
            raise D.BackendUnavailable("stub down")
        self.extracts.append((system, user))
        return json.dumps({
            "concepts": [{"label": "Widget lubrication", "kind": "concept",
                          "summary": "widgets need regular oiling",
                          "evidence": "the manual says to oil the widget"}],
            "relations": [],
            "procedures": [{"title": "Oil the widget", "goal": "smooth motion",
                            "steps": ["open the cover", "apply oil"]}],
            "criteria": []})

    def chat_json(self, system, user, schema, max_tokens=512):
        if self.down:
            raise D.BackendUnavailable("stub down")
        if "gloss" in (schema.get("properties") or {}):
            self.glosses.append((system, user))
            return {"gloss": "A maintenance manual covering widget care."}
        return None                      # any other LM ask (reconcile etc.): abstain

    def extract_narrative(self, chunk):
        return {}


class StubEmbedder:
    """Distinct texts get distinct deterministic vectors — identical vectors
    would light up the node-merge machinery for unrelated labels."""

    def embed_one(self, text, task="document"):
        import hashlib
        h = hashlib.sha256(text.encode("utf-8", "replace")).digest()
        v = [b / 255.0 + 0.01 for b in h[:8]]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]

    def embed_many(self, texts, task="document"):
        return [self.embed_one(t, task) for t in texts]


def _chunks(doc, title, n):
    return [{"id": f"{doc}#c{i}", "source_type": "reference", "title": title,
             "section": f"Section {i}", "path_or_url": doc,
             "text": f"{title} body text {i}: oil the widget gently.",
             "tokens": 12} for i in range(n)]


def main():
    tmp = Path(tempfile.mkdtemp(prefix="distiller-test-"))
    cfg = dict(khconfig.DEFAULTS)
    cfg["db_path"] = str(tmp / "raw.db")
    cfg["kb_path"] = str(tmp / "kb.db")
    store = make_store(cfg)
    kb = KB(cfg)
    store.add_chunks(_chunks("doc://widgets", "Widget manual", 3)
                     + _chunks("doc://gears", "Gear handbook", 2))
    lm, emb = StubLM(), StubEmbedder()

    stats = driver.run(store, kb, lm, emb, cfg, log_fn=lambda m: None)
    check("all pending chunks distilled across both docs",
          stats["chunks"] == 5 and stats["docs"] == 2 and stats["backend"] == "ok")
    check("one gloss call per document, not per chunk", len(lm.glosses) == 2)
    check("cards written through vinur's factory (its content-hash dedupe "
          "collapses the stub's identical procedure)", kb.db.execute(
              "SELECT count(*) FROM procedure_cards WHERE status='active'"
          ).fetchone()[0] == 1)
    check("nodes written", kb.db.execute(
        "SELECT count(*) FROM nodes WHERE status='active'").fetchone()[0] >= 1)

    sys_prompts = {s for s, _u in lm.extracts}
    check("system prompt byte-stable across chunks (cacheable prefix)",
          len(sys_prompts) == 1)
    widget_users = [u for _s, u in lm.extracts if "Widget manual" in u]
    check("digest block prepended and shared by a document's chunks",
          len(widget_users) == 3
          and all(u.startswith("=== DOCUMENT CONTEXT") for u in widget_users)
          and len({u.split("=== EXCERPT")[0] for u in widget_users}) == 1)
    check("gloss and outline reach the prompt",
          "maintenance manual" in widget_users[0]
          and "Section 0" in widget_users[0])
    ch0 = dict(store.chunks_for_path("doc://widgets")[0])
    check("stock user prompt preserved verbatim as the suffix",
          widget_users[0].endswith(D._user_prompt(ch0)))

    d, hit = digest.build(store, kb, "doc://widgets", lm=lm)
    check("digest cached in doc_meta (signature hit, no re-gloss)",
          hit is True and len(lm.glosses) == 2 and d["chunks"] == 3)

    again = driver.run(store, kb, lm, emb, cfg, log_fn=lambda m: None)
    check("checkpoint holds: rerun distils nothing",
          again["chunks"] == 0 and again["skipped"] == 5)

    store.add_chunks(_chunks("doc://late", "Late doc", 1))
    lm.down = True
    dead = driver.run(store, kb, lm, emb, cfg, log_fn=lambda m: None)
    check("dead backend stops cleanly and keeps the chunk pending",
          dead["backend"].startswith("unavailable")
          and not kb.is_distilled("doc://late#c0"))
    lm.down = False

    # ── dedup sweep: three near-duplicates + one distinct on the same node ────
    node = kb.db.execute("SELECT id FROM nodes WHERE status='active'").fetchone()["id"]
    sup = json.dumps([{"doc_id": "doc://widgets"}])
    sup2 = json.dumps([{"doc_id": "doc://gears"}])
    now = time.time()
    rows = [("card:a", "Oil the Widget", '["open the cover","apply oil"]', sup, 100.0),
            ("card:b", "oil   the widget", '["open the  cover","APPLY OIL"]', sup2, 200.0),
            ("card:c", "Oil the widget", '["open the cover","apply oil"]', sup, 300.0),
            ("card:d", "Replace the widget", '["remove","fit new"]', sup, 400.0)]
    for cid, title, steps, support, ts in rows:
        kb.db.execute("INSERT INTO procedure_cards(id,node_id,title,steps,support,"
                      "status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                      (cid, node, title, steps, support, "active", ts, ts))
    kb.db.commit()

    active_before = kb.db.execute("SELECT count(*) FROM procedure_cards "
                                  "WHERE status='active'").fetchone()[0]
    dry = merge.sweep(kb, dry=True, log_fn=lambda m: None)
    check("dry run reports without changing anything",
          dry["merged"] >= 2 and kb.db.execute(
              "SELECT count(*) FROM procedure_cards WHERE status='active'"
          ).fetchone()[0] == active_before)
    res = merge.sweep(kb, log_fn=lambda m: None)
    surv = kb.db.execute("SELECT * FROM procedure_cards WHERE id='card:a'").fetchone()
    check("normalized duplicates (case/whitespace) merge into the oldest",
          res["merged"] >= 2
          and kb.db.execute("SELECT status FROM procedure_cards WHERE id='card:b'"
                            ).fetchone()["status"] == "merged"
          and kb.db.execute("SELECT status FROM procedure_cards WHERE id='card:c'"
                            ).fetchone()["status"] == "merged")
    check("survivor accumulates observed_count across the group",
          surv["observed_count"] >= 3)
    check("survivor support is the union of the group's docs",
          {e["doc_id"] for e in json.loads(surv["support"])}
          >= {"doc://widgets", "doc://gears"})
    check("distinct card untouched", kb.db.execute(
        "SELECT status FROM procedure_cards WHERE id='card:d'"
    ).fetchone()["status"] == "active")

    kb.close()
    if FAILED:
        print(f"\n{len(FAILED)} FAILED")
        raise SystemExit(1)
    print("\nALL OK")


if __name__ == "__main__":
    main()
