"""Tests for tool_facade: the simplified fast-LM tool surface (routing, filtering, introspection)."""
import asyncio

import tool_facade as tf


class FakeInner:
    """A stand-in ToolHost: advertises a fixed catalogue, records calls."""
    def __init__(self, names, active=True):
        self._names = names
        self._active = active
        self.calls = []
    @property
    def active(self):
        return self._active
    async def catalogue(self):
        return [{"type": "function", "function": {"name": n, "description": n,
                 "parameters": {"type": "object", "properties": {}}}} for n in self._names]
    async def call(self, name, arguments):
        self.calls.append((name, arguments)); return f"ok:{name}"
    async def call_raw(self, name, arguments):
        self.calls.append((name, arguments)); return {"ok": True, "result": f"ok:{name}", "error": ""}


FULL = ["mail_list", "mail_recent", "mail_search", "mail_read",
        "file_search", "file_read", "file_list", "file_index",
        "calendar_today", "calendar_range", "calendar_create", "calendar_update", "calendar_delete",
        "weather", "news_headlines", "fourchan_catalog", "fourchan_thread",
        "web_search", "web_fetch", "kb_search", "kb_ask"]


def run(coro): return asyncio.run(coro)


# ── pure routing ────────────────────────────────────────────────────────────────

def test_resolve_by_action():
    mail = next(w for w in tf.WRAPPERS if w["name"] == "mail")
    assert tf.resolve(mail, {"action": "search", "query": "tax"}) == ("mail_search", {"query": "tax"})
    assert tf.resolve(mail, {"action": "read", "id": "7"}) == ("mail_read", {"id": "7"})


def test_resolve_default_when_action_unknown():
    mail = next(w for w in tf.WRAPPERS if w["name"] == "mail")
    assert tf.resolve(mail, {"action": "weird"})[0] == "mail_recent"   # falls to default
    assert tf.resolve(mail, {})[0] == "mail_recent"


def test_wrapper_targets():
    web = next(w for w in tf.WRAPPERS if w["name"] == "web")
    assert set(tf.wrapper_targets(web)) >= {"web_search", "web_fetch"}


# ── catalogue shaping ───────────────────────────────────────────────────────────

def test_catalogue_collapses_and_passes_through():
    fac = tf.FacadeHost(FakeInner(FULL))
    names = [(t.get("function") or {}).get("name") for t in run(fac.catalogue())]
    # wrappers present
    for w in ("mail", "files", "news", "web", "calendar_read"):
        assert w in names, w
    # granular members hidden
    for hidden in ("mail_search", "file_read", "news_headlines", "web_search", "calendar_range"):
        assert hidden not in names, hidden
    # passthrough kept native
    for keep in ("weather", "kb_search", "kb_ask", "calendar_create", "calendar_delete"):
        assert keep in names, keep
    # noisy internals dropped entirely
    for gone in ("file_index", "fourchan_thread", "mail_list"):
        assert gone not in names, gone


def test_wrapper_omitted_when_no_underlying_available():
    # Only knowledge tools exist → no mail/files/etc. wrappers, but kb_* pass through.
    fac = tf.FacadeHost(FakeInner(["kb_search", "kb_ask"]))
    names = [(t.get("function") or {}).get("name") for t in run(fac.catalogue())]
    assert names == ["kb_search", "kb_ask"] or set(names) == {"kb_search", "kb_ask"}
    assert "mail" not in names and "web" not in names


# ── call dispatch ───────────────────────────────────────────────────────────────

def test_call_routes_wrapper_to_primitive():
    inner = FakeInner(FULL)
    fac = tf.FacadeHost(inner)
    run(fac.call("files", {"action": "read", "path": "/a"}))
    assert inner.calls[-1] == ("file_read", {"path": "/a"})


def test_call_passthrough_and_internal_names_untouched():
    inner = FakeInner(FULL)
    fac = tf.FacadeHost(inner)
    run(fac.call("kb_search", {"query": "x"}))            # passthrough
    run(fac.call_raw("calendar_range", {"days": 5}))      # internal direct call (not catalogued)
    assert ("kb_search", {"query": "x"}) in inner.calls
    assert ("calendar_range", {"days": 5}) in inner.calls


def test_news_folds_forum_to_fourchan():
    inner = FakeInner(FULL)
    fac = tf.FacadeHost(inner)
    run(fac.call("news", {"source": "forum", "query": "ai"}))
    assert inner.calls[-1] == ("fourchan_catalog", {"query": "ai"})
    run(fac.call("news", {}))                              # default → headlines
    assert inner.calls[-1] == ("news_headlines", {})


# ── introspection / tracing ──────────────────────────────────────────────────────

def test_trace_reports_availability_once_then_locks_on():
    events = []
    fac = tf.FacadeHost(FakeInner(FULL), trace=events.append)
    run(fac.catalogue()); run(fac.catalogue())            # second is unchanged → no duplicate
    facs = [e for e in events if e["kind"] == "tool_facade"]
    assert len(facs) == 1
    assert "mail" in facs[0]["offered"] and "file_index" in facs[0]["hidden"]
    run(fac.call("mail", {"action": "search", "query": "q"}))
    locks = [e for e in events if e["kind"] == "tool_locked"]
    assert locks and locks[-1]["wrapper"] == "mail" and locks[-1]["tool"] == "mail_search"


def test_active_proxies_inner():
    assert tf.FacadeHost(FakeInner([], active=False)).active is False
    assert tf.FacadeHost(FakeInner([], active=True)).active is True


def main():
    import types
    p = f = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and isinstance(fn, types.FunctionType):
            try:
                fn(); p += 1; print(f"  ok  {name}")
            except Exception as e:
                f += 1; print(f"FAIL  {name}: {e}")
    print(f"\n{p} passed, {f} failed")
    raise SystemExit(1 if f else 0)


if __name__ == "__main__":
    main()
