"""PDF — text layer via **PyMuPDF** (`fitz`), fast and layout-aware, with an
**OCR fallback** for scanned pages.

Sectioning prefers the document outline (TOC) when present; otherwise pages are
grouped under their nearest preceding TOC entry, or emitted page-by-page.  OCR
is the *exception*, not the rule: only pages whose text layer yields almost
nothing are rendered and run through ``ocrmypdf``/``tesseract`` (per-page, via
subprocess on a temp single-page PDF — no network, parser kept constrained).
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile

from . import MissingDependency

log = logging.getLogger("knowledgehost.sources.pdf")

_errors_muted = False


def _fitz():
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise MissingDependency(f"PyMuPDF (fitz) required for PDFs — "
                                f"run ./install.sh --pdf to add it ({e})")
    # Malformed-but-recoverable PDFs make MuPDF spew "syntax error: …" lines straight to
    # stderr (from C, bypassing logging) — a firehose on a big crawl.  Silence it ONCE,
    # process-wide; we drain the accumulated warnings per-file into our own debug log
    # instead, so they stay visible at log_level=DEBUG without drowning a normal run.
    global _errors_muted
    if not _errors_muted:
        try:
            fitz.TOOLS.mupdf_display_errors(False)
        except Exception:                      # older/newer API — best-effort, never fatal
            pass
        _errors_muted = True
    return fitz


def _drain_warnings(fitz, path: str) -> None:
    """Clear MuPDF's accumulated warnings for the file just parsed and note the count at
    DEBUG — recoverable syntax warnings mean 'repaired', not 'failed', so they're not
    worth an INFO/WARN line each, but the count is handy when a specific PDF misbehaves."""
    try:
        w = fitz.TOOLS.mupdf_warnings(reset=True)
    except Exception:
        return
    if w:
        n = sum(1 for line in str(w).splitlines() if line.strip())
        log.debug("%s: %d recoverable MuPDF warning(s) (text still extracted)",
                  os.path.basename(path), n)


def _ocr_page(src_path: str, page_no: int) -> str:
    """OCR a single page via ocrmypdf, returning its extracted text (or "")."""
    fitz = _fitz()
    try:
        with tempfile.TemporaryDirectory() as td:
            one = os.path.join(td, "p.pdf")
            out = os.path.join(td, "o.pdf")
            doc = fitz.open(src_path)
            sub = fitz.open()
            sub.insert_pdf(doc, from_page=page_no, to_page=page_no)
            sub.save(one)
            sub.close(); doc.close()
            r = subprocess.run(
                ["ocrmypdf", "--quiet", "--force-ocr", "--optimize", "0", one, out],
                capture_output=True, timeout=180)
            if r.returncode != 0:
                return ""
            d = fitz.open(out)
            text = d[0].get_text("text")
            d.close()
            return text or ""
    except (subprocess.SubprocessError, OSError, RuntimeError) as e:
        log.debug("OCR failed on page %d: %s", page_no, e)
        return ""


def extract(path: str, cfg: dict):
    fitz = _fitz()
    doc = fitz.open(path)                       # a truly unopenable file raises → caught upstream
    blocks = []
    skipped = 0
    try:
        try:
            meta_title = (doc.metadata or {}).get("title") or ""
        except Exception:
            meta_title = ""
        title = meta_title.strip() or os.path.splitext(os.path.basename(path))[0]

        # TOC: list of [level, title, page] -> nearest heading per page
        try:
            toc = doc.get_toc() or []
        except Exception:                      # a broken outline shouldn't cost us the text
            toc = []
        page_heading = {}
        if toc:
            stack = []
            for level, htitle, page in toc:
                stack[:] = stack[:level - 1] + [htitle.strip()]
                page_heading[max(0, page - 1)] = " > ".join([h for h in stack if h])

        cur_heading = ""
        min_chars = cfg.get("ocr_min_chars", 32)
        want_ocr = cfg.get("ocr", True)
        try:
            page_count = doc.page_count
        except Exception:
            page_count = 0
        # Load pages by index and guard each one: a single malformed page (the kind that
        # raises rather than merely warning) is skipped, not allowed to sink the whole doc.
        for i in range(page_count):
            if i in page_heading:
                cur_heading = page_heading[i]
            try:
                text = doc.load_page(i).get_text("text") or ""
            except Exception as e:
                log.debug("%s: unreadable page %d skipped (%s)",
                          os.path.basename(path), i, e)
                skipped += 1
                continue
            if want_ocr and len(text.strip()) < min_chars:
                try:
                    ocr = _ocr_page(path, i)
                    if len(ocr.strip()) > len(text.strip()):
                        text = ocr
                except Exception:
                    pass
            if text.strip():
                blocks.append((cur_heading, text))
    finally:
        try:
            doc.close()
        except Exception:
            pass
        _drain_warnings(fitz, path)
    if skipped:
        log.info("%s: %d page(s) unreadable, skipped; kept %d block(s)",
                 os.path.basename(path), skipped, len(blocks))
    return title, blocks
