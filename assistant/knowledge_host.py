"""
Thin async client for the standalone **knowledge-host** (a separate app on 127.0.0.1).

The knowledge-host is Vinkona's metacognitive/procedural knowledge base: a hybrid
(dense + FTS) retriever over distilled "cards" and "nodes", with an intent-conditioned
reranker and a grounded-answer path.  It speaks a small HTTP contract:

    POST /call  {name, arguments}  ->  {ok, result: <json-string>}    (the tool result)
    GET  /health                   ->  liveness + index stats

Two tools matter here:
  • kb_ask    {query, rigor}          -> DISTILLED, grounded items (concepts, relations,
                                          *procedures*) with provenance + a confidence band.
                                          This is the metacognitive layer — "what to DO with
                                          a thing", not its glossary definition.
  • kb_search {query, k, intent}      -> cited passages + a rerank confidence; `intent`
                                          is local-only and focuses the ranking.

This client is deliberately fail-soft: any error, timeout, or unreachable host returns
None, never raises.  It's called from the *background* big-LM briefing path (never the
voice critical path), so a slow or down host can only mean "no guidance this turn".

Security: localhost only (the knowledge-host binds 127.0.0.1, never the LAN).  If the
host sets an auth_token, pass it through as a Bearer token.
"""

import asyncio
import json
import typing as tp

import aiohttp


class KnowledgeHost:
    """Async client over the knowledge-host's POST /call contract."""

    def __init__(self, url: str, *, token: str = "", timeout_s: float = 4.0):
        self.url = (url or "").rstrip("/")
        self.token = token or ""
        self.timeout = aiohttp.ClientTimeout(total=max(0.5, float(timeout_s)))

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    async def _call(self, name: str, arguments: dict,
                    http: tp.Optional[aiohttp.ClientSession] = None) -> tp.Optional[dict]:
        """POST /call and return the parsed tool result dict, or None on any failure.

        The host wraps the tool's own JSON in {ok, result:<json-string>}; we unwrap both
        layers so the caller gets the tool's native result (e.g. {passages, confidence})."""
        if not self.url:
            return None
        payload = json.dumps({"name": name, "arguments": arguments}).encode()
        own = http is None
        sess = http or aiohttp.ClientSession()
        try:
            async with sess.post(f"{self.url}/call", data=payload,
                                 headers=self._headers(), timeout=self.timeout) as resp:
                if resp.status != 200:
                    return None
                outer = await resp.json()
            if not outer.get("ok"):
                return None
            result = outer.get("result")
            if isinstance(result, str):           # tool result is JSON-encoded inside `result`
                try:
                    return json.loads(result)
                except (ValueError, json.JSONDecodeError):
                    return None
            return result if isinstance(result, dict) else None
        except (aiohttp.ClientError, OSError, ValueError, json.JSONDecodeError,
                asyncio.TimeoutError):
            return None
        finally:
            if own:
                await sess.close()

    async def ask(self, query: str, *, rigor: str = "low",
                  context_features: tp.Optional[dict] = None, intent: str = "",
                  http: tp.Optional[aiohttp.ClientSession] = None) -> tp.Optional[dict]:
        """Grounded, distilled answer (kb_ask) — concepts/relations/procedures + confidence.

        context_features ({feature: value} discriminators — what triggered it, the setting,
        the observable) are what make kb_ask accurate: the host scores answers by how well
        they match and ABSTAINS on a mismatch, so supplying them is what stops a near-topic
        wrong card.  intent (how/why_diag/why_mech/what) focuses a noun-phrase query."""
        query = (query or "").strip()
        if not query:
            return None
        args: dict = {"query": query}
        if rigor and rigor != "low":
            args["rigor"] = rigor
        if context_features:
            args["context_features"] = context_features
        if intent:
            args["intent"] = intent
        return await self._call("kb_ask", args, http=http)

    async def search(self, query: str, *, intent: str = "", k: int = 5,
                     http: tp.Optional[aiohttp.ClientSession] = None) -> tp.Optional[dict]:
        """Cited passages (kb_search), with the local-only `intent` focusing the rerank."""
        query = (query or "").strip()
        if not query:
            return None
        args: dict = {"query": query, "k": int(k)}
        if intent:
            args["intent"] = intent
        return await self._call("kb_search", args, http=http)
