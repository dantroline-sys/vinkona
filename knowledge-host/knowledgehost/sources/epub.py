"""EPUB — iterate document items via **ebooklib**, run each (HTML) chapter
through the shared HTML sectioner.  Chapter file order is the reading order.
"""
from __future__ import annotations

import os

from . import MissingDependency
from . import html as html_src


def extract(path: str, cfg: dict):
    try:
        import ebooklib
        from ebooklib import epub
    except Exception as e:
        raise MissingDependency(f"ebooklib required for EPUB — "
                                f"run ./install.sh --epub to add it ({e})")

    book = epub.read_epub(path)
    title = ""
    md = book.get_metadata("DC", "title")
    if md:
        title = md[0][0]
    title = (title or "").strip() or os.path.splitext(os.path.basename(path))[0]

    blocks = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        try:
            raw = item.get_content().decode("utf-8", "replace")
        except Exception:
            continue
        _t, chapter_blocks = html_src.extract_from_string(raw)
        blocks.extend(chapter_blocks)
    return title, blocks
