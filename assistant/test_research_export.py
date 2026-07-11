"""Tests for research_export — turning the non-personal research hoard (documents) into <hash>.md
drops for the knowledge host.  Exercises the personal-content firewall, plan-question mapping,
idempotent writes, the incremental watermark, and full re-export (repairing removed files)."""
import os
import sqlite3
import tempfile

import research_export as rx


class FakeMem:
    """Just the bits research_export needs: a db + a KV state store."""
    def __init__(self):
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.db.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, url TEXT, title TEXT, "
                        "topic TEXT, fetched_at REAL, text TEXT, digest TEXT, kind TEXT DEFAULT 'research')")
        self.db.execute("CREATE TABLE worker_state (key TEXT PRIMARY KEY, value TEXT)")
        self._n = 0
    def add(self, topic, text, *, title="via", url="", digest="", kind="research", fetched=None):
        self._n += 1
        self.db.execute("INSERT INTO documents(id,url,title,topic,fetched_at,text,digest,kind) "
                        "VALUES (?,?,?,?,?,?,?,?)",
                        (f"d{self._n}", url, title, topic, fetched or float(self._n), text, digest, kind))
        self.db.commit()
    def get_state(self, key, default=None):
        r = self.db.execute("SELECT value FROM worker_state WHERE key=?", (key,)).fetchone()
        return r["value"] if r else default
    def set_state(self, key, value):
        self.db.execute("INSERT INTO worker_state(key,value) VALUES(?,?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))
        self.db.commit()


def _tmp():
    return tempfile.mkdtemp(prefix="rxtest-")


def _files(folder):
    return sorted(f for f in os.listdir(folder) if f.endswith(".md"))


# ── personal-content firewall ─────────────────────────────────────────────────────

def test_crawl_and_denylisted_topics_never_exported():
    m = FakeMem()
    m.add("dissociation in psychiatry", "research body", kind="research")
    m.add("mail-inbox", "PRIVATE EMAIL BODY", kind="crawl")               # personal (kind)
    m.add("files-documents", "PRIVATE FILE TEXT", kind="research")        # legacy crawl (topic denylist)
    folder = _tmp()
    res = rx.export_research(m, folder, crawl_sources=["mail-inbox", "files-documents"], full=True)
    assert res["questions"] == 1 and res["written"] == 1
    blob = "".join(open(os.path.join(folder, f)).read() for f in _files(folder))
    assert "research body" in blob
    assert "PRIVATE EMAIL BODY" not in blob and "PRIVATE FILE TEXT" not in blob


# ── question mapping (research topic vs plan question) ─────────────────────────────

def test_plan_question_uses_title_not_generic_topic():
    m = FakeMem()
    m.add("plan", "answer to A", title="What is condition A?")            # plan doc: question in title
    m.add("plan", "answer to B", title="What is condition B?")
    folder = _tmp()
    res = rx.export_research(m, folder, full=True)
    # two distinct plan questions → two files (not merged under the generic 'plan' bucket)
    assert res["questions"] == 2 and len(_files(folder)) == 2
    a = open(os.path.join(folder, rx.question_hash("What is condition A?") + ".md")).read()
    assert "# Question" in a and "What is condition A?" in a       # question under the # Question head


# ── one file per question, sources accumulate ─────────────────────────────────────

def test_same_question_groups_into_one_file():
    m = FakeMem()
    m.add("theory of mind in autism", "source one text", url="http://a")
    m.add("Theory Of Mind  in autism", "source two text", url="http://b")  # same q (case/space)
    folder = _tmp()
    res = rx.export_research(m, folder, full=True)
    assert res["questions"] == 1 and len(_files(folder)) == 1
    doc = open(os.path.join(folder, _files(folder)[0])).read()
    assert "source one text" in doc and "source two text" in doc          # both sources present
    assert "http://a" in doc and "http://b" in doc


# ── idempotency + incremental watermark + full repair ─────────────────────────────

def test_idempotent_second_run_skips_unchanged():
    m = FakeMem()
    m.add("q one", "body")
    folder = _tmp()
    assert rx.export_research(m, folder, full=True)["written"] == 1
    r2 = rx.export_research(m, folder, full=True)
    assert r2["written"] == 0 and r2["skipped"] == 1                      # byte-identical → skip


def test_incremental_only_new_since_watermark():
    m = FakeMem()
    m.add("first q", "b1")
    folder = _tmp()
    assert rx.export_research(m, folder, full=False)["written"] == 1      # exports first, sets watermark
    m.add("second q", "b2")
    r = rx.export_research(m, folder, full=False)
    assert r["written"] == 1 and r["questions"] == 2                      # only the new question written
    assert set(_files(folder)) == {rx.question_hash("first q") + ".md",
                                   rx.question_hash("second q") + ".md"}


def test_full_reexport_repairs_deleted_file():
    m = FakeMem()
    m.add("keep me", "body")
    folder = _tmp()
    rx.export_research(m, folder, full=False)
    os.remove(os.path.join(folder, rx.question_hash("keep me") + ".md"))  # user deleted it
    # incremental won't notice (watermark passed it); full rebuilds everything
    assert rx.export_research(m, folder, full=False)["written"] == 0
    assert rx.export_research(m, folder, full=True)["written"] == 1
    assert _files(folder) == [rx.question_hash("keep me") + ".md"]


def test_no_folder_is_a_soft_error():
    m = FakeMem()
    m.add("q", "b")
    res = rx.export_research(m, "", full=True)
    assert res["ok"] is False and "folder" in res["error"]


def test_render_matches_host_parser_format():
    # Must match knowledgehost/research.py: provenance: vinkona, `# Question`, `## Answer`, `## Sources`.
    doc = rx.render_doc("what is X", [{"rowid": 1, "url": "http://s", "title": "Wikipedia",
                                       "text": "X is a thing", "digest": "X summary", "fetched_at": 1.0}])
    assert doc.startswith("---")
    assert "provenance: vinkona\n" in doc and "trust: low" in doc     # the host's is_research_doc gate
    assert "# Question\n\nwhat is X" in doc                         # question under the literal heading
    assert "## Answer" in doc and "X summary" in doc
    assert "## Sources" in doc and "### Wikipedia" in doc and "X is a thing" in doc


def main():
    import types
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and isinstance(fn, types.FunctionType):
            try:
                fn(); passed += 1; print(f"  ok  {name}")
            except Exception as e:
                failed += 1; print(f"FAIL  {name}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
