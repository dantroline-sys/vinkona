"""CLI — run the enhanced distill pass and/or the dedup sweep against a host.

    python3 -m distiller --config /path/to/config.toml [--vinur /path/to/vinur]
                         [--bundle B] [--limit N] [--no-gloss]
                         [--merge | --merge-only] [--dry]

Run from the vinkona repo root (or with it on PYTHONPATH).  Uses the host's
own config for kb/store/LM/embedder, so it points wherever the host points.
Disable the stock distill entries in the host's autopilot ops while this is
the driver (see driver.py's operational rule).
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(prog="distiller", description=__doc__.splitlines()[0])
    ap.add_argument("--vinur", default=os.environ.get("VINUR_REPO",
                                                      "/home/user/vinur"))
    ap.add_argument("--config", default=os.environ.get("KNOWLEDGE_CONFIG"))
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-gloss", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="run the dedup sweep after distilling")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--dry", action="store_true", help="merge: report, change nothing")
    args = ap.parse_args()

    from . import bootstrap, driver, merge
    bootstrap(args.vinur)
    from knowledgehost.config import load_config
    from knowledgehost.distill import DistillLM
    from knowledgehost.embed import Embedder
    from knowledgehost.kb import KB
    from knowledgehost.store import make_store

    cfg = load_config(args.config)
    kb = KB(cfg)
    try:
        if not args.merge_only:
            store = make_store(cfg)
            stats = driver.run(store, kb, DistillLM(cfg), Embedder(cfg), cfg,
                               bundle=args.bundle, limit=args.limit,
                               use_gloss=not args.no_gloss, log_fn=print)
            print(json.dumps({"distill": stats}, ensure_ascii=False))
        if args.merge or args.merge_only:
            res = merge.sweep(kb, bundle=args.bundle, dry=args.dry, log_fn=print)
            print(json.dumps({"merge": res}, ensure_ascii=False))
            if res["merged"] and not args.dry:
                print("note: run the host's rebuild-fts — cards_fts does not "
                      "track status flips")
    finally:
        kb.close()


if __name__ == "__main__":
    main()
