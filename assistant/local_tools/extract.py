"""File → text for the local files genre.

Plain text and code by decoding; Office/OpenDocument formats by walking the
XML inside their zip containers (stdlib only — a .docx is just a zip); HTML by
tag-stripping; PDF via pypdf when it is installed (optional — a clear message
otherwise, never a silent empty read).  Everything returns (ok, text) where a
False carries the reason instead of the text.
"""
import html
import io
import re
import zipfile
import xml.etree.ElementTree as ET

TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".log", ".tex", ".org", ".py", ".js", ".ts",
    ".dart", ".rs", ".c", ".h", ".cpp", ".hpp", ".java", ".kt", ".go", ".rb",
    ".sh", ".bash", ".sql", ".xml", ".svg", ".css", ".scss", ".php", ".pl",
    ".lua", ".r", ".swift", ".m", ".vb", ".ps1", ".bat", ".eml", ".ics", ".vcf",
}

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?i)</(p|div|br|li|h[1-6]|tr)[^>]*>", "\n", text)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", _WS.sub(" ", text)).strip()


def _decode(raw: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _xml_texts(raw: bytes, para_tags: tuple) -> str:
    """All text nodes of an XML document, with a newline at each closing tag
    whose LOCAL name is in para_tags (namespaces vary by producer)."""
    out = []
    try:
        for ev, el in ET.iterparse(io.BytesIO(raw), events=("end",)):
            local = el.tag.rsplit("}", 1)[-1]
            if el.text and el.text.strip():
                out.append(el.text)
            if local in para_tags:
                out.append("\n")
    except ET.ParseError:
        pass
    return re.sub(r"\n{3,}", "\n\n", "".join(out)).strip()


def _from_zip(path: str, members: tuple, para_tags: tuple) -> str:
    chunks = []
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        for pattern in members:
            for n in sorted(names):
                if re.fullmatch(pattern, n):
                    chunks.append(_xml_texts(z.read(n), para_tags))
    return "\n".join(c for c in chunks if c)


def _pdf(path: str) -> tuple:
    try:
        import pypdf
    except ImportError:
        return (False, "PDF reading needs the optional pypdf package "
                       "(pip install pypdf into vinkona_env) — or keep PDFs on "
                       "a Mac tool host, whose reader also does OCR")
    try:
        reader = pypdf.PdfReader(path)
        pages = []
        for page in reader.pages[:200]:
            pages.append(page.extract_text() or "")
        text = "\n".join(pages).strip()
        return (True, text) if text else (False, "no extractable text (a scanned PDF needs OCR)")
    except Exception as e:
        return False, f"could not read PDF: {e}"


def extract_text(path: str, max_chars: int = 20000) -> tuple:
    """(ok, text-or-reason) for one file, bounded to max_chars."""
    low = path.lower()
    try:
        if low.endswith(".pdf"):
            ok, text = _pdf(path)
        elif low.endswith(".docx"):
            ok, text = True, _from_zip(path, (r"word/document\.xml",), ("p",))
        elif low.endswith(".odt"):
            ok, text = True, _from_zip(path, (r"content\.xml",), ("p", "h"))
        elif low.endswith(".xlsx"):
            ok, text = True, _from_zip(
                path, (r"xl/sharedStrings\.xml", r"xl/worksheets/sheet\d+\.xml"), ("si", "row"))
        elif low.endswith(".pptx"):
            ok, text = True, _from_zip(path, (r"ppt/slides/slide\d+\.xml",), ("p", "sp"))
        elif low.endswith((".html", ".htm")):
            with open(path, "rb") as f:
                ok, text = True, _strip_html(_decode(f.read(4 * 1024 * 1024)))
        elif low.endswith(".rtf"):
            with open(path, "rb") as f:
                raw = _decode(f.read(4 * 1024 * 1024))
            text = re.sub(r"\\[a-z]+-?\d* ?|[{}]|\\'[0-9a-f]{2}", "", raw)
            ok = True
        else:
            with open(path, "rb") as f:
                raw = f.read(4 * 1024 * 1024)
            if b"\x00" in raw[:4096] and not low.endswith(tuple(TEXT_SUFFIXES)):
                return False, "binary file — no text to extract"
            ok, text = True, _decode(raw)
    except FileNotFoundError:
        return False, "file not found"
    except PermissionError:
        return False, "permission denied"
    except zipfile.BadZipFile:
        return False, "corrupt or unsupported document container"
    except Exception as e:
        return False, f"could not read: {e}"
    if not ok:
        return False, text
    text = text.strip()
    if not text:
        return False, "the file has no extractable text"
    return True, text[: max(200, int(max_chars))]
