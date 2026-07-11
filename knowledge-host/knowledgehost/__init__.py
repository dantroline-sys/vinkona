"""Vinkona Knowledge Host — a standalone local general-knowledge tool host.

A large, local, searchable knowledge base Vinkona can query mid-conversation: a
Wikipedia snapshot plus the user's own PDFs, books, journals and miscellany,
returning *cited* passages.  It speaks the standard Vinkona tool-host contract
(`GET /tools` + `POST /call`, see ../assistant/MAC_TOOLS.md) so it is just another
host in Vinkona's `MultiHost` — point `tools.knowledge.url` at it and the fast LM
can call `kb_search` like any other tool.

Two halves with very different shapes (see ../assistant/KNOWLEDGE.md):
- an **offline ingestion pipeline** (`ingest`)  — heavy, batch, run on demand;
- a **query service** (`server`)               — light, fast, always up.

It is a **separate store from Vinkona's `memories`**: bulk, low-trust, reference-
only, with its own ANN/FTS index.  Returns data, never instructions; every
passage is sanitized + cited before any LM reads it.
"""

__version__ = "1.0.0"
