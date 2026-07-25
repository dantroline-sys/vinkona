#!/usr/bin/env python
"""
Tests for the in-process knowledge tier (local_kb.LocalKB + choose_backend).

Runs on a bare interpreter against the PAIRED vinur checkout (../vinur — the
same sibling convention the module itself uses).  No LM, no embedder, no host:
the point of the tier is that vinur's read path works as a plain library, so
the tests exercise exactly that — build a tiny .kdb pack with vinur's own
producer machinery, import it through LocalKB, and ask/search in-process.

    python test_local_kb.py          (prints SKIP and exits 0 without a vinur checkout)
"""

import asyncio
import inspect
import json
import os
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).parent
VINUR = HERE.resolve().parent.parent / "vinur"

PASS = 0


def ok(label):
    global PASS
    PASS += 1
    print(f"  ok {PASS:2d}  {label}")


def main():
    if not (VINUR / "knowledgehost").is_dir():
        print("SKIP: no paired vinur checkout at", VINUR)
        return 0

    sys.path.insert(0, str(HERE))
    import local_kb as LK

    # ── a broken vinur path fails soft: disabled + reason, never a raise ────
    lk_bad = LK.LocalKB({"vinur_path": "/nonexistent/vinur"})
    assert not lk_bad.enabled and "vinur_path" in lk_bad.reason, lk_bad.reason
    assert asyncio.run(lk_bad.ask("anything")) is None
    assert asyncio.run(lk_bad.search("anything")) is None
    ok("unavailable vinur: enabled=False, reason set, ask/search fail-soft None")

    # ── duck-type parity with the HTTP client (aiohttp stubbed to import it) ─
    sys.modules.setdefault("aiohttp", types.SimpleNamespace(
        ClientTimeout=lambda **k: None, ClientSession=None, ClientError=Exception))
    import knowledge_host as KH
    for meth in ("ask", "search", "swap_in"):
        want = set(inspect.signature(getattr(KH.KnowledgeHost, meth)).parameters)
        have = set(inspect.signature(getattr(LK.LocalKB, meth)).parameters)
        assert want <= have, (meth, want - have)
    assert hasattr(LK.LocalKB, "enabled")
    ok("LocalKB duck-types KnowledgeHost: ask/search/swap_in signatures cover the client's")

    with tempfile.TemporaryDirectory() as td:
        # ── produce a tiny pack with vinur's own machinery (clean-room style) ─
        sys.path.insert(0, str(VINUR))
        from knowledgehost import bundles as B
        from knowledgehost import config as KHC
        from knowledgehost.kb import KB

        prod_cfg = dict(KHC.DEFAULTS)
        prod_cfg.update({"kb_path": os.path.join(td, "prod", "kb.db"),
                         "db_path": os.path.join(td, "prod", "index.db"),
                         "bundle_dir": os.path.join(td, "prod", "bundles"),
                         "embed_url": ""})
        prod = KB(prod_cfg)
        sup = json.dumps([{"doc_id": "doc:rocks", "trust": 0.9}])
        prod.db.execute(
            "INSERT INTO source_registry(doc_id,title,source_type,trust_weight,status,bundle)"
            " VALUES('doc:rocks','Rocks: A Primer','book',0.9,'active','geology')")
        prod.db.execute(
            "INSERT INTO nodes(id,label,kind,summary,aliases,support,status,embedding)"
            " VALUES('n_rock','rock','entity','a naturally occurring solid aggregate of "
            "minerals','[]',?, 'active',NULL)", (sup,))
        prod.db.commit()
        res = B.split(prod_cfg, force=True)
        prod.close()
        assert "geology" in res, res
        pack = res["geology"]["file"]

        # ── consumer: LocalKB over a VIRGIN kb path, explicit vinur_path ─────
        lk = LK.LocalKB({"vinur_path": str(VINUR),
                         "kb_path": os.path.join(td, "cons", "kb.db"),
                         "index_path": os.path.join(td, "cons", "index.db")})
        assert lk.enabled, lk.reason
        ok("LocalKB constructs against an explicit vinur checkout (virgin kb path)")

        r1 = lk.import_pack(pack)
        assert r1["sources_new"] == 1, r1
        assert os.path.exists(os.path.join(td, "cons", "kb.db"))
        r2 = lk.import_pack(pack)
        assert r2["sources_new"] == 0, "re-import must be a no-op"
        ok("import_pack: first import bootstraps + lands the pack; re-import no-op")

        c = lk.counts()
        assert c.get("nodes", 0) == 1, c
        ok("counts() sees the imported knowledge after the post-import reopen")

        # ── ask/search in-process, embedder-less (lexical tier) ─────────────
        bundle = asyncio.run(lk.ask("what is a rock?"))
        assert isinstance(bundle, dict), bundle
        assert "confidence" in bundle or "abstain" in bundle or "items" in bundle, bundle
        ok("ask(): kb_ask answers in-process — client-shaped bundle dict, no embedder")

        sr = asyncio.run(lk.search("rock"))
        assert sr is None or isinstance(sr, dict)
        assert asyncio.run(lk.ask("")) is None, "empty query short-circuits"
        assert asyncio.run(lk.swap_in("big")) is None, "nothing to swap in-process"
        ok("search()/swap_in(): shape-safe over an empty chunk store; empty query None")

        # ── the tier ladder: host wins, local falls in, unavailable -> None ──
        made = {}

        class FakeKH:
            def __init__(self, url, token="", timeout_s=4.0):
                made["url"] = url

        kh_mod = types.SimpleNamespace(KnowledgeHost=FakeKH)
        cfg_host = {"knowledge_host": {"enabled": True, "url": "http://h:1"},
                    "local_kb": {"enabled": True}}
        assert isinstance(LK.choose_backend(cfg_host, kh_mod), FakeKH)
        assert made["url"] == "http://h:1"
        cfg_local = {"knowledge_host": {"enabled": False},
                     "local_kb": {"enabled": True, "vinur_path": str(VINUR),
                                  "kb_path": os.path.join(td, "cons", "kb.db"),
                                  "index_path": os.path.join(td, "cons", "index.db")}}
        picked = LK.choose_backend(cfg_local, kh_mod)
        assert isinstance(picked, LK.LocalKB) and picked.enabled
        picked.close()
        cfg_none = {"knowledge_host": {"enabled": False},
                    "local_kb": {"enabled": True, "vinur_path": "/nonexistent"}}
        assert LK.choose_backend(cfg_none, kh_mod) is None, \
            "an unavailable local tier must yield None, not a dead backend"
        assert LK.choose_backend({}, kh_mod) is None
        ok("choose_backend ladder: host > local > None; unavailable local -> None")

        lk.close()

    print(f"test_local_kb: {PASS} checks OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
