"""
VIN-TOOL-01 TG4: the flow engine — executes pinned graphs, and runs the §6
self-test protocol before anything is trusted.

Execution is deliberately dumb: steps arrive already in execution order (the
validator pinned them), every value crossing an edge is shape-checked against
the DECLARED semantic type at runtime, and each step leaves a provenance row
(§6 anti-invisibility: counts in, counts out, which step ate what).  A failing
step stops the run with its evidence; nothing downstream sees partial data.

Ctx implementations here:
  LiveCtx       real side effects — injected callables for net/llm, a clock,
                and WRITE CONTAINMENT: store writes resolve inside the store
                root or fail; dry mode redirects them to a scratch dir and
                snapshots nothing real (there is nothing real to snapshot);
                live mode snapshots a file before first overwrite (§0.5).
  RecordingCtx  wraps any ctx and records net/llm traffic, so a live run
                leaves behind the replay data the NEXT self-test's shadow
                pass uses (§6.2 "replayed from recorded fixtures where
                possible").

selftest() = §6 stages 1–3: static revalidation, the fixture pass (each
constituent block's own fixtures, plus an end-to-end shadow run when recorded
traffic exists), and the dry-run with mechanical appraisal — an optional
`appraise` hook receives the evidence and returns her own reading of it, which
the caller logs as an aside (§0.6).  Probation/deploy bookkeeping is the graph
store's job (TG5).

Pure stdlib.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import blocks as _blocks
import toolgraph as _tg
from blocks import BlockError, Ctx, ReplayCtx


# ── Ctx implementations ───────────────────────────────────────────────────────
class LiveCtx(Ctx):
    """Real side effects, all injected — the engine owns policy, not sockets.
    `net`/`llm` are callables (the broker-backed fetcher and the big LM lane
    get wired in at the worker); `store_dir` is the ONLY writable root."""

    def __init__(self, *, net=None, llm=None, now=time.time,
                 store_dir: str | Path | None = None,
                 dry: bool = False, scratch_dir: str | Path | None = None):
        self._net, self._llm, self._now = net, llm, now
        self.store_dir = Path(store_dir) if store_dir else None
        self.dry = bool(dry)
        self.scratch_dir = Path(scratch_dir) if scratch_dir else None
        self.snapshots: dict = {}         # store-relative path -> snapshot Path
        self.writes: list = []            # store-relative paths written this run

    def net(self, url: str) -> str:
        if self._net is None:
            raise BlockError("this graph has no network access")
        return self._net(url)

    def llm(self, prompt: str) -> str:
        if self._llm is None:
            raise BlockError("this graph has no language-model access")
        return self._llm(prompt)

    def now(self) -> float:
        return self._now()

    def write(self, relpath: str, text: str) -> str:
        """The one write faculty (§0.5).  Containment is enforced HERE, not
        trusted to blocks: the resolved target must stay inside the root."""
        root = self.scratch_dir if self.dry else self.store_dir
        if root is None:
            raise BlockError("no writable store for this run"
                             + (" (dry mode needs a scratch dir)" if self.dry else ""))
        root = root.resolve()
        target = (root / relpath).resolve()
        if root != target and root not in target.parents:
            raise BlockError(f"write escapes the store: {relpath!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not self.dry and target.exists() and relpath not in self.snapshots:
            snap = target.with_suffix(target.suffix + ".snap")
            shutil.copy2(target, snap)
            self.snapshots[relpath] = snap
        target.write_text(text)
        self.writes.append(relpath)
        return str(target)

    def rollback(self):
        """Restore every snapshotted file; remove files this run created."""
        if self.dry or self.store_dir is None:
            return
        root = self.store_dir.resolve()
        for relpath in self.writes:
            target = (root / relpath).resolve()
            snap = self.snapshots.get(relpath)
            if snap and Path(snap).exists():
                shutil.copy2(snap, target)
            elif target.exists():
                target.unlink()


class RecordingCtx(Ctx):
    """Wrap a ctx; remember net/llm traffic as replay data for later shadow
    passes.  Writes pass straight through to the wrapped ctx's containment."""

    def __init__(self, inner: Ctx):
        self.inner = inner
        self.recorded: dict = {"net": {}, "llm": {}}

    def net(self, url: str) -> str:
        out = self.inner.net(url)
        self.recorded["net"][url] = out
        return out

    def llm(self, prompt: str) -> str:
        out = self.inner.llm(prompt)
        self.recorded["llm"][prompt] = out
        return out

    def now(self) -> float:
        return self.inner.now()

    def write(self, relpath: str, text: str) -> str:
        return self.inner.write(relpath, text)


# ── The executor ──────────────────────────────────────────────────────────────
def _count(v) -> int:
    return len(v) if isinstance(v, list) else 1


def run_graph(pinned: dict, ctx: Ctx) -> dict:
    """Execute a PINNED graph.  Returns {"ok", "outputs", "steps", "error"};
    "steps" is the provenance trail — one row per step, kept even on failure
    so 'which step ate everything' is answerable in seconds (§6)."""
    slots: dict = {}
    trail: list = []
    for s in pinned.get("steps", []):
        b = _blocks.get(s["block"], s.get("block_version"))
        row = {"id": s["id"], "block": b.key, "ok": False,
               "in": {}, "out": {}, "ms": 0}
        trail.append(row)
        try:
            inputs = {}
            for port, want in b.ports_in.items():
                if port not in (s.get("inputs") or {}):
                    raise BlockError(f"input {port!r} is unbound — run the "
                                     "validator before the executor")
                bound = s["inputs"][port]
                ref = _tg.parse_ref(bound) if isinstance(bound, str) else None
                if ref and f"{ref[0]}.{ref[1]}" not in slots:
                    raise BlockError(f"{bound!r} was never produced — run the "
                                     "validator before the executor")
                value = slots[f"{ref[0]}.{ref[1]}"] if ref else bound
                bad = _blocks.check_value(want, value)
                if bad:
                    raise BlockError(f"input {port!r}: {bad}")
                inputs[port] = value
                row["in"][port] = _count(value)
            t0 = time.monotonic()
            got = b.fn(inputs, dict(s.get("params") or {}), ctx)
            row["ms"] = int((time.monotonic() - t0) * 1000)
            for port, t in b.ports_out.items():
                if port not in (got or {}):
                    raise BlockError(f"block returned no {port!r}")
                bad = _blocks.check_value(t, got[port])
                if bad:
                    raise BlockError(f"output {port!r}: {bad}")
                slots[f"{s['id']}.{port}"] = got[port]
                row["out"][port] = _count(got[port])
            row["ok"] = True
        except BlockError as e:
            row["error"] = str(e)
            return {"ok": False, "outputs": None, "steps": trail,
                    "error": f"step {s['id']} ({b.name}): {e}"}
    outputs = {}
    for name, bound in (pinned.get("outputs") or {}).items():
        ref = _tg.parse_ref(bound)
        outputs[name] = slots.get(f"{ref[0]}.{ref[1]}") if ref else None
    return {"ok": True, "outputs": outputs, "steps": trail, "error": ""}


# ── The §6 self-test protocol, stages 1–3 ─────────────────────────────────────
def _mechanical_appraisal(result: dict) -> list:
    """Honest observations, not verdicts: the classic invisible failure is a
    step that quietly ate everything (§6 exclusion-sampling spirit)."""
    notes = []
    for row in result.get("steps", []):
        fed = sum(n for n in row["in"].values())
        made = sum(n for n in row["out"].values())
        if row["ok"] and fed > 0 and made == 0:
            notes.append(f"step {row['id']} ({row['block']}) received {fed} "
                         "item(s) and produced none — check its settings")
    return notes


def selftest(pinned: dict, live_ctx: LiveCtx | None = None, *,
             recorded: dict | None = None, appraise=None) -> dict:
    """Static pass → fixture pass (block fixtures + recorded shadow compose
    when possible) → live dry-run.  Returns {"passed", "stages", "notes",
    "appraisal", "recorded"} — on failure the evidence is Gap-Report material.
    `appraise` (optional callable(evidence_text) -> str) is her own reading of
    the dry-run, logged by the caller as an aside; it never decides the verdict
    alone — a self-test cannot pass on vibes."""
    stages: dict = {}
    notes: list = []

    # 1 — static
    v = _tg.validate(pinned)
    stages["static"] = {"ok": v.ok, "errors": list(v.errors)}
    if not v.ok:
        return {"passed": False, "stages": stages, "notes": notes,
                "appraisal": "", "recorded": None}

    # 2 — fixtures: every constituent block's own fixtures still pass…
    fx_fail: list = []
    for s in pinned["steps"]:
        b = _blocks.get(s["block"], s.get("block_version"))
        for fx in b.fixtures:
            ok, detail = _blocks.run_fixture(b, fx)
            if not ok:
                fx_fail.append(detail)
    # …and, when recorded traffic exists, the whole graph composes in shadow.
    shadow = None
    if recorded:
        shadow = run_graph(pinned, ReplayCtx(recorded))
        if not shadow["ok"]:
            fx_fail.append(f"shadow compose: {shadow['error']}")
    stages["fixtures"] = {"ok": not fx_fail, "errors": fx_fail,
                          "shadow_composed": bool(recorded)}
    if fx_fail:
        return {"passed": False, "stages": stages, "notes": notes,
                "appraisal": "", "recorded": None}

    # 3 — live dry-run (skipped without a live ctx: emission-time callers may
    # run stages 1–2 cheaply and defer the dry-run to deploy time)
    if live_ctx is None:
        stages["dry_run"] = {"ok": None, "skipped": True}
        return {"passed": True, "stages": stages, "notes": notes,
                "appraisal": "", "recorded": None}
    caps = {c for s in pinned["steps"]
            for c in _blocks.get(s["block"], s.get("block_version")).capabilities}
    live_ctx.dry = True
    if caps & {"mutate", "fs-write"} and live_ctx.scratch_dir is None:
        stages["dry_run"] = {"ok": False, "errors":
                             ["a mutate/fs-write graph needs a scratch dir for its dry-run"]}
        return {"passed": False, "stages": stages, "notes": notes,
                "appraisal": "", "recorded": None}
    rec = RecordingCtx(live_ctx)
    result = run_graph(pinned, rec)
    notes = _mechanical_appraisal(result)
    stages["dry_run"] = {"ok": result["ok"],
                         "errors": [result["error"]] if result["error"] else [],
                         "steps": result["steps"]}
    appraisal = ""
    if appraise is not None:
        evidence = [f"goal: {pinned.get('goal')}", f"ran ok: {result['ok']}"]
        for row in result["steps"]:
            evidence.append(f"  {row['id']} {row['block']}: in {row['in']} "
                            f"out {row['out']}" + (f" ERROR {row.get('error')}"
                                                   if not row["ok"] else ""))
        evidence += [f"note: {n}" for n in notes]
        try:
            appraisal = str(appraise("\n".join(evidence)) or "")
        except Exception as e:
            appraisal = f"(appraisal unavailable: {e})"
    return {"passed": bool(result["ok"]), "stages": stages, "notes": notes,
            "appraisal": appraisal,
            "recorded": rec.recorded if result["ok"] else None}
