"""Per-format extractors.  Each returns ``(title, blocks)`` where ``blocks`` is
an iterable of ``(section_path, text)`` — a heading path ("History > Founding")
and the prose under it.  chunk.py turns those into storable chunks.

Heavy parsers (PyMuPDF, ebooklib, libzim, trafilatura) are imported lazily
inside each extractor, so a format the user hasn't installed support for simply
raises ``MissingDependency`` for that file and the crawl carries on.
"""
from __future__ import annotations

import os


class MissingDependency(RuntimeError):
    pass


def extractor_for(path: str):
    """Return the extractor callable for a path's extension, or None."""
    ext = os.path.splitext(path)[1].lower()
    from . import pdf, epub, html, text
    return {
        ".pdf": pdf.extract,
        ".epub": epub.extract,
        ".html": html.extract, ".htm": html.extract,
        ".txt": text.extract, ".md": text.extract,
    }.get(ext)
