"""
VIN-TOOL-01 TG5: the deployed-graph store — §6.4 probation and §7 change
control, as files.

One JSON document per deployed graph under <own_tools root>/graphs/.  A graph
deploys into PROBATION (§6.4): its first N real runs keep full provenance and
count down; a failure while on probation auto-disables the graph and hands the
caller a §8-shaped Gap Report to file.  N successes promote it to ACTIVE.
Failures of an active graph are recorded but do not disable — a proven tool
failing usually means the world moved, not the tool.

§7/§0.7 change control: deploy snapshots the sha256 of every block the graph
uses; load_runnable() recomputes and REFUSES a mismatch — a changed block
never silently changes a deployed tool.  Redeploying a name resets probation
(an upgraded graph re-earns trust, §7).

The store lives inside the own-tools root — the same write-containment
boundary as everything else she owns.  Pure stdlib.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import blocks as _blocks
import toolgraph as _tg

STATES = ("probation", "active", "disabled")
_RUNS_KEPT = 10


class GraphStoreError(Exception):
    pass


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


class GraphStore:
    def __init__(self, root: str | Path, *, probation_runs: int = 5):
        self.dir = Path(root) / "graphs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.probation_runs = max(1, int(probation_runs))

    def _path(self, name: str) -> Path:
        safe = "".join(c for c in str(name) if c.isalnum() or c == "_")
        if not safe:
            raise GraphStoreError("bad graph name")
        return self.dir / f"{safe}.json"

    def _write(self, doc: dict) -> dict:
        p = self._path(doc["name"])
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1))
        tmp.replace(p)
        return doc

    # ── deploy / read ────────────────────────────────────────────────────────
    def deploy(self, pinned: dict, *, recorded: dict | None = None,
               appraisal: str = "") -> dict:
        """Register a self-tested graph.  Validates once more (deploys are
        rare; paranoia is cheap), snapshots block hashes, starts probation."""
        v = _tg.validate(pinned)
        if not v.ok:
            raise GraphStoreError(f"refusing to deploy an invalid graph: {v.errors[:3]}")
        used = {b.key for b in v.resolved.values()}
        current = _blocks.manifest()
        doc = {"name": pinned["name"], "goal": pinned.get("goal", ""),
               "graph": v.pinned, "state": "probation",
               "probation_left": self.probation_runs,
               "capabilities": sorted(v.capabilities),
               "capability_summary": _tg.capability_summary(v),
               "needs_approval": _tg.needs_approval(v),
               "manifest": {k: current[k] for k in sorted(used)},
               "recorded": recorded or None, "appraisal": str(appraisal or ""),
               "created_at": _now_iso(), "schedule_s": 0,
               "runs": [], "run_count": 0, "disabled_reason": ""}
        return self._write(doc)

    def get(self, name: str) -> dict | None:
        try:
            return json.loads(self._path(name).read_text())
        except (OSError, ValueError):
            return None

    def list(self) -> list:
        out = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                d = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            out.append({k: d.get(k) for k in
                        ("name", "goal", "state", "probation_left",
                         "capabilities", "capability_summary", "needs_approval",
                         "created_at", "schedule_s", "run_count",
                         "disabled_reason", "appraisal")}
                       | {"last_run": (d.get("runs") or [None])[-1]})
        return out

    def load_runnable(self, name: str) -> tuple:
        """(pinned, recorded) — refusing a disabled graph or a block whose
        source no longer matches the deploy-time hash (§7)."""
        doc = self.get(name)
        if doc is None:
            raise GraphStoreError(f"no deployed graph named {name!r}")
        if doc["state"] == "disabled":
            raise GraphStoreError(
                f"'{name}' is disabled: {doc.get('disabled_reason') or 'by hand'}")
        current = _blocks.manifest()
        drift = sorted(k for k, h in doc.get("manifest", {}).items()
                       if current.get(k) != h)
        if drift:
            raise GraphStoreError(
                f"blocks changed since '{name}' was deployed ({', '.join(drift)}) "
                "— redeploy to re-test it against the new blocks")
        return doc["graph"], doc.get("recorded")

    # ── probation bookkeeping (§6.4) ─────────────────────────────────────────
    def record_run(self, name: str, result: dict) -> dict:
        """Append a run summary and apply the probation rules.  Returns
        {"doc", "gap"} — gap is a §8-shaped report the CALLER files when a
        probation run regressed (this store keeps books; it doesn't queue)."""
        doc = self.get(name)
        if doc is None:
            raise GraphStoreError(f"no deployed graph named {name!r}")
        summary = {"ok": bool(result.get("ok")), "at": _now_iso(),
                   "at_s": time.time(),
                   "error": (result.get("error") or "")[:400],
                   "steps": [{"id": r.get("id"), "block": r.get("block"),
                              "in": r.get("in"), "out": r.get("out"),
                              **({"excluded": r["excluded"],
                                  "excluded_sample": r["excluded_sample"]}
                                 if r.get("excluded") else {}),
                              **({"error": r["error"]} if r.get("error") else {})}
                             for r in (result.get("steps") or [])]}
        doc["runs"] = (doc.get("runs") or [])[-(_RUNS_KEPT - 1):] + [summary]
        doc["run_count"] = int(doc.get("run_count") or 0) + 1
        gap = None
        if doc["state"] == "probation":
            if summary["ok"]:
                doc["probation_left"] = max(0, int(doc.get("probation_left") or 0) - 1)
                if doc["probation_left"] == 0:
                    doc["state"] = "active"
            else:
                doc["state"] = "disabled"
                doc["disabled_reason"] = f"regressed on probation: {summary['error']}"
                gap = {"kind": "defect", "goal": doc.get("goal", ""),
                       "attempted_graph": doc["name"],
                       "missing": "",
                       "closest_existing": [s["block"] for s in summary["steps"]],
                       "evidence": summary["error"],
                       "urgency": "quality"}
        self._write(doc)
        return {"doc": doc, "gap": gap}

    # ── hand controls ────────────────────────────────────────────────────────
    def set_state(self, name: str, state: str, reason: str = "") -> dict:
        if state not in STATES:
            raise GraphStoreError(f"bad state {state!r}")
        doc = self.get(name)
        if doc is None:
            raise GraphStoreError(f"no deployed graph named {name!r}")
        doc["state"] = state
        doc["disabled_reason"] = reason if state == "disabled" else ""
        if state == "probation":
            doc["probation_left"] = self.probation_runs
        return self._write(doc)

    def set_schedule(self, name: str, schedule_s: int) -> dict:
        doc = self.get(name)
        if doc is None:
            raise GraphStoreError(f"no deployed graph named {name!r}")
        doc["schedule_s"] = max(0, int(schedule_s))
        return self._write(doc)

    def remove(self, name: str) -> bool:
        p = self._path(name)
        if not p.exists():
            return False
        p.unlink()
        return True
