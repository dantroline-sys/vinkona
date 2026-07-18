"""Distillation add-in for the vinur knowledge host.

The oleum-grade pipeline upgrades (OLEUM-DST-01 discipline) applied to text
distillation: per-document context digests, prefix-cache-friendly prompting,
and a dedup sweep that accumulates repeat observations into observed_count
instead of minting duplicate cards.

Lives in the vinkona repo so it stays out of vinur's Apache-2.0 tree (this
repo is PolyForm Noncommercial).  vinur is consumed strictly as a library
through documented call surfaces — the add-in can be deleted and the stock
distill verb resumes unchanged.
"""
__version__ = "0.1.0"


def bootstrap(vinur_repo):
    """Put the vinur checkout on sys.path; fail fast if it isn't one."""
    import sys
    from pathlib import Path
    p = str(Path(vinur_repo).expanduser().resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
    import knowledgehost  # noqa: F401 — wrong path should fail here, not later
