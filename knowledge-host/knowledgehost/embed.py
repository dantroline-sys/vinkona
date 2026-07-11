"""Embedding client for the nomic endpoint (OpenAI /v1/embeddings, llama.cpp
--embedding mode), shared with Vinkona's memory store for one consistent vector
space.

Asymmetric model => task prefixes: the **document** side is prefixed at ingest
(`search_document: `), the **query** side at search time (`search_query: `).
Stdlib-only (urllib) so the ingestion pipeline and the threaded HTTP server need
no async runtime.  Returns L2-normalized float32 vectors, so cosine == dot.

If the endpoint is unreachable the calls return None and the host degrades to
sparse-only FTS retrieval (logged once) rather than failing.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

log = logging.getLogger("knowledgehost.embed")


class Embedder:
    def __init__(self, cfg: dict):
        self.url = cfg["embed_url"].rstrip("/")
        self.model = cfg["embed_model"]
        self.use_prefix = bool(cfg["embed_task_prefix"])
        self.qpfx = cfg["embed_query_prefix"]
        self.dpfx = cfg["embed_document_prefix"]
        self.timeout = cfg["embed_timeout_s"]
        self.max_tokens = cfg.get("embed_max_tokens", 512)
        # Tokens are estimated as chars/4 (no tokenizer here); clip on chars.
        self._max_chars = self.max_tokens * 4 if self.max_tokens else 0
        self._warned = False
        self._trunc_warned = False
        self._reject_warned = False
        self.dim: int | None = None

    def _prefix(self, text: str, task: str) -> str:
        if not self.use_prefix:
            return text
        return (self.qpfx if task == "query" else self.dpfx) + text

    def _truncate(self, text: str) -> str:
        """Clip an already-prefixed input to the embed model's window.

        The server (llama.cpp --embedding) processes each sequence in a single
        physical batch; an input over its n_ubatch (~embed_max_tokens) tokens is
        rejected with HTTP 500.  A stray oversized chunk is embedded from its
        first N tokens rather than nulling the whole batch into sparse-only.
        """
        if self._max_chars and len(text) > self._max_chars:
            if not self._trunc_warned:
                log.warning("input over embed window (~%d tokens) — embedding the "
                            "first ~%d tokens", len(text) // 4, self.max_tokens)
                self._trunc_warned = True
            return text[:self._max_chars]
        return text

    def _post(self, payload: dict):
        req = urllib.request.Request(
            f"{self.url}/v1/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def _normalize(self, vec):
        # Stdlib L2 normalize (numpy optional; this path has no numpy dependency).
        import math
        n = math.sqrt(sum(x * x for x in vec))
        if n <= 0:
            return None
        out = [x / n for x in vec]
        self.dim = len(out)
        return out

    def embed_one(self, text: str, task: str = "document"):
        """One vector as a python list[float], or None if the endpoint is down."""
        out = self.embed_many([text], task)
        return out[0] if out else None

    def embed_many(self, texts: list[str], task: str = "document"):
        """Batch embed.  Returns list[list[float] | None] aligned with `texts`,
        or None for the whole batch **only** on a transport failure (endpoint
        down).

        A server-side rejection (HTTP 500 — typically a single input over the
        embed model's physical-batch window, n_ubatch) does not sink the batch:
        we bisect to isolate the offender and embed it from its head, so at most
        that one chunk falls back to sparse rather than all `embed_batch` of
        them.  This makes ingestion robust without the client knowing the
        server's exact token limit."""
        if not texts:
            return []
        inputs = [self._truncate(self._prefix(t, task)) for t in texts]
        return self._embed_inputs(inputs)

    def _embed_inputs(self, inputs: list[str]):
        try:
            data = self._post({"model": self.model, "input": inputs}).get("data") or []
        except urllib.error.HTTPError as e:
            # The endpoint is up but rejected this request — almost always one
            # input over n_ubatch (or the whole batch over n_batch).  With more
            # than one input, bisect to find the offender; a lone rejected input
            # is retried shorter so it keeps a head-truncated vector.
            if len(inputs) == 1:
                return [self._embed_shrinking(inputs[0], e)]
            mid = len(inputs) // 2
            left = self._embed_inputs(inputs[:mid])
            right = self._embed_inputs(inputs[mid:])
            if left is None or right is None:
                return None                  # a half hit a transport failure
            return left + right
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as e:
            if not self._warned:
                log.warning("embed endpoint unreachable (%s) — running sparse-only", e)
                self._warned = True
            return None
        # llama.cpp preserves input order; index defensively anyway.
        out: list = [None] * len(inputs)
        for item in data:
            i = item.get("index", 0)
            if 0 <= i < len(out):
                out[i] = self._normalize(item.get("embedding") or [])
        return out

    def _embed_shrinking(self, text: str, err):
        """A single input the server rejected: trim the head and retry so the
        chunk still gets a (truncated) vector.  None only if it fails even when
        heavily trimmed, or the endpoint drops mid-retry."""
        for _ in range(5):
            text = text[: int(len(text) * 0.8)]
            if not text:
                break
            try:
                data = self._post({"model": self.model, "input": [text]}).get("data") or []
            except urllib.error.HTTPError:
                continue                     # still too long — trim further
            except (urllib.error.URLError, OSError, ValueError, TimeoutError):
                return None
            if data:
                return self._normalize(data[0].get("embedding") or [])
        if not self._reject_warned:
            log.warning("embed rejected a chunk even after trimming (%s) — "
                        "sparse-only for it", err)
            self._reject_warned = True
        return None
