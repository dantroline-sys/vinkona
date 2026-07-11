"""Single-slot operations runner for the web control panel.

The maintenance verbs (ingest/distill/link/refine/…) are launched as **subprocesses** of
the long-lived server — the very same `python3 -m knowledgehost <verb>` you'd run by hand.
The running job is tracked by holding its live `Popen` in memory: "is a job running?" is
answered by the **kernel** (`proc.poll()` is None while alive, the exit code once done), so
there is no lock file to go stale when something crashes mid-run.  One job at a time (they
contend on the GPU lease and the KB); a second request is refused while one is live.

Safety: only the allow-listed verbs below can be launched, and only with their typed
options — the UI sends `{limit: 20, fast: true}`, never a command string — so there is no
shell-injection surface (subprocess list form, no `shell=True`).
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger("knowledgehost.ops")

# verb -> {option: type}.  type ∈ int | bool | choice:<a,b>.  The flag is --<option-with-dashes>.
COMMANDS: dict = {
    "ingest":     {"force": "bool", "wikipedia": "bool", "distill": "bool", "limit": "int"},
    "ingest-library": {"force": "bool"},   # index the search-only document library (library_sources)
    "rebuild-fts": {},                     # reindex FTS with the configured tokenizer (no re-parse)
    "distill":    {"limit": "int", "watch": "bool", "interval": "int"},
    "link":       {"limit": "int", "fast": "bool", "top_k": "int"},
    "refine":     {"limit": "int", "force": "bool"},
    "adjudicate": {"limit": "int", "batch": "int", "fast": "bool",
                   "no_auto": "bool", "auto_only": "bool"},
    "reconcile":  {"limit": "int", "top_k": "int", "anchors": "choice:corpus,all"},
    "build-ann":  {},
    "embed-nodes": {"limit": "int"},
    "optimize":   {"vacuum": "bool"},
    "stats":      {},
    "split":      {"force": "bool"},    # export each bundle group to its <bundle>.kdb file
}


def _argv(command: str, args: dict) -> list:
    """Validate the typed options for `command` and render them to CLI flags.  Raises
    ValueError on an unknown verb/option/value — nothing unvalidated reaches the shell."""
    if command not in COMMANDS:
        raise ValueError(f"unknown command: {command}")
    spec = COMMANDS[command]
    out: list = []
    for key, val in (args or {}).items():
        if key not in spec:
            raise ValueError(f"{command}: unknown option {key!r}")
        typ = spec[key]
        flag = "--" + key.replace("_", "-")
        if typ == "bool":
            if val:
                out.append(flag)
        elif typ == "int":
            out += [flag, str(int(val))]               # int() rejects non-numeric
        elif typ.startswith("choice:"):
            allowed = typ.split(":", 1)[1].split(",")
            sv = str(val)
            if sv not in allowed:
                raise ValueError(f"{command}: {key} must be one of {allowed}")
            out += [flag, sv]
    return out


class OpsRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.config_path = cfg.get("_config_path")
        ctrl = cfg.get("control_dir") or str(Path(__file__).resolve().parent.parent / "var")
        self.logdir = Path(ctrl).expanduser() / "ops-logs"
        self.logdir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._job: dict | None = None              # the live job, in memory — the source of truth

    def _build_cmd(self, command: str, argv: list) -> list:
        cmd = [sys.executable, "-m", "knowledgehost"]
        if self.config_path:                            # same config the server runs on
            cmd += ["-c", self.config_path]
        return cmd + [command, *argv]

    def running(self) -> bool:
        j = self._job
        return bool(j and j["proc"].poll() is None)

    def start(self, command: str, args: dict | None = None) -> dict:
        with self._lock:
            if self.running():
                return {"ok": False, "error": "a job is already running", "status": self.status()}
            argv = _argv(command, args or {})          # raises on anything invalid
            ts = time.strftime("%Y%m%d-%H%M%S")
            logfile = self.logdir / f"{command}-{ts}.log"
            lf = open(logfile, "wb", buffering=0)
            cmd = self._build_cmd(command, argv)
            log.info("ops: launching %s", " ".join(cmd))
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}   # stream prints to the log live
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    start_new_session=True,   # own group → clean tree kill
                                    env=env)
            self._job = {"proc": proc, "logfh": lf, "command": command, "argv": argv,
                         "started": time.time(), "logfile": str(logfile)}
            return {"ok": True, "status": self.status()}

    def stop(self) -> dict:
        with self._lock:
            if not self.running():
                return {"ok": False, "error": "no job is running"}
            proc = self._job["proc"]
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)   # whole session
            except (ProcessLookupError, PermissionError):
                proc.terminate()
        for _ in range(20):                            # up to ~2s for a graceful exit
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
        return {"ok": True, "status": self.status()}

    def status(self) -> dict:
        j = self._job
        if not j:
            return {"running": False, "command": None}
        rc = j["proc"].poll()
        return {"running": rc is None, "command": j["command"], "argv": j["argv"],
                "started": j["started"], "elapsed_s": round(time.time() - j["started"]),
                "exit_code": rc, "logfile": j["logfile"]}

    def tail(self, n: int = 300) -> str:
        j = self._job
        if not j:
            return ""
        try:
            with open(j["logfile"], "r", errors="replace") as f:
                return "".join(f.readlines()[-int(n):])
        except OSError:
            return ""

    def shutdown(self) -> None:
        """Best-effort: stop a running job when the server itself is going down, so a job
        doesn't outlive the server that was tracking it."""
        if self.running():
            log.info("ops: server shutdown — stopping the running job")
            self.stop()
