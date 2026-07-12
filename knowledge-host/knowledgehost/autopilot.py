"""
Autopilot — run maintenance verbs automatically, in a user-defined priority order.

The knowledge host has several background verbs (ingest, distill, link, adjudicate,
refine, …).  Left to a human they get run by hand; left to a dumb timer they run in
the wrong order.  The autopilot runs them on a priority basis: the ORDER of the step
list is the priority, and after every step it re-evaluates from the top — so a
high-priority step that just gained work (e.g. fresh Vinkona research drops to
distil) preempts the big uncurated backlog below it.

Why this order matters (the motivating case): Vinkona's own research outputs are
more immediately relevant than mountains of ingested PDFs, so 'distil the vinkona
bundle' sits above 'distil everything else' — the former drains fully (it's small),
the latter runs in bounded batches so the former can preempt it when new drops land.

Coordination: the single-slot OpsRunner means one job at a time (a manual job from
the Operations tab always wins).  And when the assistant is doing its own idle work
it holds the LM leases (lm_fast / lm_big); with respect_leases on, the autopilot
stands down so the two never fight over the GPUs — the mirror of the assistant's
'pause idle work' button.

Config lives in var/autopilot.json (live-editable from the Prioritizer tab); this
module owns its defaults and the pure step-selection logic (unit-tested).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

log = logging.getLogger("knowledgehost.autopilot")

# The verbs the autopilot may schedule, and the sensible default plan.  Distill is
# split so Vinkona's drops go first; the long backlog steps carry a limit so each
# invocation is bounded and the loop can re-check priorities between batches.
DEFAULT_PLAN = {
    "enabled": False,               # opt-in; the Prioritizer tab turns it on
    "idle_interval_s": 60,          # wait this long when a full pass found no work
    "respect_leases": True,         # stand down while the assistant holds the LMs
    "steps": [
        {"command": "ingest",     "args": {},                    "enabled": True,
         "min_interval_s": 900,   "label": "Ingest new/changed documents"},
        {"command": "distill",    "args": {"bundle": "vinkona"}, "enabled": True,
         "min_interval_s": 0,     "label": "Distil Vinkona's research drops (priority)"},
        {"command": "distill",    "args": {"limit": 50},         "enabled": True,
         "min_interval_s": 0,     "label": "Distil the rest of the corpus (batched)"},
        {"command": "link",       "args": {"limit": 200},        "enabled": True,
         "min_interval_s": 3600,  "label": "Link related cards"},
        {"command": "adjudicate", "args": {"limit": 100},        "enabled": True,
         "min_interval_s": 3600,  "label": "Merge duplicate nodes"},
        {"command": "refine",     "args": {"limit": 50},         "enabled": False,
         "min_interval_s": 86400, "label": "Refine cards against their sources"},
    ],
}


def plan_path(cfg: dict) -> Path:
    root = Path(__file__).resolve().parent.parent
    ctrl = cfg.get("control_dir") or str(root / "var")
    return Path(ctrl).expanduser() / "autopilot.json"


def load_plan(cfg: dict) -> dict:
    """The saved plan, or the default.  Missing keys are backfilled so an older file
    keeps working after we add a field."""
    p = plan_path(cfg)
    plan = json.loads(json.dumps(DEFAULT_PLAN))     # deep copy
    try:
        if p.exists():
            saved = json.loads(p.read_text())
            for k in ("enabled", "idle_interval_s", "respect_leases"):
                if k in saved:
                    plan[k] = saved[k]
            if isinstance(saved.get("steps"), list):
                plan["steps"] = saved["steps"]
    except Exception as e:                          # pragma: no cover
        log.warning("autopilot: bad plan file %s (%s) — using defaults", p, e)
    return plan


def save_plan(cfg: dict, plan: dict) -> dict:
    """Validate and persist a plan from the UI.  Only known commands/fields survive."""
    from .ops import COMMANDS
    clean_steps = []
    for s in plan.get("steps", []):
        cmd = str(s.get("command", ""))
        if cmd not in COMMANDS:
            continue
        clean_steps.append({
            "command": cmd,
            "args": s.get("args") if isinstance(s.get("args"), dict) else {},
            "enabled": bool(s.get("enabled", True)),
            "min_interval_s": max(0, int(s.get("min_interval_s", 0) or 0)),
            "label": str(s.get("label", cmd))[:120],
        })
    out = {
        "enabled": bool(plan.get("enabled", False)),
        "idle_interval_s": max(5, int(plan.get("idle_interval_s", 60) or 60)),
        "respect_leases": bool(plan.get("respect_leases", True)),
        "steps": clean_steps or DEFAULT_PLAN["steps"],
    }
    p = plan_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")
    return out


def due_step(steps: list, last_run: dict, now: float):
    """Pure selection: the first enabled step (highest priority) whose min_interval has
    elapsed since it last ran.  Returns (index, step) or (None, None).

    `last_run` maps a step key -> unix ts of its last completion.  Re-evaluating from
    the top each call is what gives priority preemption."""
    for i, s in enumerate(steps):
        if not s.get("enabled", True):
            continue
        key = step_key(s)
        gap = now - last_run.get(key, 0.0)
        if gap >= float(s.get("min_interval_s", 0) or 0):
            return i, s
    return None, None


def step_key(step: dict) -> str:
    """Stable identity for a step (command + its args) so last-run tracking survives
    reordering in the UI."""
    args = step.get("args") or {}
    return step["command"] + "(" + ",".join(f"{k}={args[k]}" for k in sorted(args)) + ")"


class Autopilot:
    """Background driver: picks due steps in priority order and runs them through the
    server's single-slot OpsRunner, yielding to manual jobs and (optionally) to the
    assistant's LM leases."""

    def __init__(self, cfg: dict, ops, lease_mod=None):
        self.cfg = cfg
        self.ops = ops
        self.lease = lease_mod
        self._last_run: dict = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = {"running_step": None, "last_reason": "not started"}

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="autopilot", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def status(self) -> dict:
        plan = load_plan(self.cfg)
        return {"enabled": plan["enabled"], "running_step": self._state["running_step"],
                "last_reason": self._state["last_reason"],
                "respect_leases": plan["respect_leases"]}

    # ── the loop ─────────────────────────────────────────────────────────────
    def _leases_held(self, plan) -> bool:
        if not plan.get("respect_leases", True) or self.lease is None:
            return False
        try:
            return (self.lease.is_held(self.lease.FAST, self.cfg)
                    or self.lease.is_held(self.lease.BIG, self.cfg))
        except Exception:                           # pragma: no cover
            return False

    def _loop(self):
        log.info("autopilot thread started")
        while not self._stop.is_set():
            try:
                plan = load_plan(self.cfg)
                if not plan.get("enabled"):
                    self._state.update(running_step=None, last_reason="disabled")
                    self._sleep(5)
                    continue
                if self.ops.running():              # a manual job owns the single slot
                    self._state["last_reason"] = "a manual job is running"
                    self._sleep(5)
                    continue
                if self._leases_held(plan):         # the assistant is using the LMs
                    self._state.update(running_step=None, last_reason="yielding — assistant holds the LMs")
                    self._sleep(min(30, plan["idle_interval_s"]))
                    continue
                idx, step = due_step(plan["steps"], self._last_run, time.time())
                if step is None:
                    self._state.update(running_step=None, last_reason="all steps idle / not due")
                    self._sleep(plan["idle_interval_s"])
                    continue
                self._run_step(step)
            except Exception as e:                  # pragma: no cover - defensive
                log.warning("autopilot loop error (continuing): %s", e)
                self._sleep(10)
        log.info("autopilot thread stopped")

    def _run_step(self, step):
        label = step.get("label", step["command"])
        self._state["running_step"] = label
        log.info("autopilot: running %s (%s)", step["command"], label)
        res = self.ops.start(step["command"], step.get("args") or {})
        if not res.get("ok"):
            # Couldn't launch (bad args, or a job slipped in) — note and back off so we
            # don't hot-loop on a broken step.
            self._last_run[step_key(step)] = time.time()
            self._state["last_reason"] = f"{step['command']}: {res.get('error','could not start')}"
            self._sleep(10)
            return
        # Wait for completion, staying responsive to stop.
        while not self._stop.is_set() and self.ops.running():
            self._stop.wait(2)
        self._last_run[step_key(step)] = time.time()
        self._state["last_reason"] = f"ran {step['command']}"
        # Loop restarts from the top → priority preemption.

    def _sleep(self, secs):
        self._stop.wait(secs)
