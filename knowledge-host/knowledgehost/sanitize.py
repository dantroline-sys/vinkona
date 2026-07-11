"""Sanitize ingested / returned text before it reaches the language model.

ALL ingested content is UNTRUSTED (Wikipedia edits, random PDFs, downloaded
books can carry prompt-injection).  The tool returns **data, never
instructions**: we strip control characters and neutralize chat/turn control
tokens so a passage can colour an answer but can never issue a command.  Vinkona
additionally fences results as low-trust (`safety.sanitize_external` +
`wrap_untrusted`) on its side — this is defence in depth, not a replacement.
"""
from __future__ import annotations

import re

# C0/C1 control chars except common whitespace (\t \n \r kept, then collapsed).
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x80-\x9f]")

# Chat-template / instruction markers various LMs honor.
_TOKENS = re.compile(
    r"""<\|[^|>]{0,40}\|>             # <|im_start|>, <|endoftext|>, …
      | <\s*/?\s*(?:s|im_start|im_end|system|user|assistant|tool)\s*>
      | \[/?\s*(?:INST|SYS|/?s)\s*\]   # [INST] [/INST] [SYS]
      | \#{2,}\s*(?:system|user|assistant|human|ai)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def clean(text, max_len: int = 0) -> str:
    """Strip control chars + turn tokens, collapse whitespace, optional truncate."""
    if not text:
        return ""
    s = str(text)
    s = _CONTROL.sub(" ", s)
    s = _TOKENS.sub(" ", s)
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s).strip()
    if max_len and len(s) > max_len:
        s = s[:max_len].rstrip() + "…"
    return s
