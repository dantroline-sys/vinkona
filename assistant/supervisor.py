#!/usr/bin/env python3
"""
Vinkona service supervisor — one stdlib-only process that owns the whole stack.

This replaces the tmux + pkill orchestration: every service is a direct child
of the supervisor (its own process group), so lifecycle is exact — no pattern
matching to find our own processes, no window/session bookkeeping.  Output
goes to logs/<name>.log (truncated per start, same as the old tee), which the
config web UI already tails.  The web UI's control protocol is unchanged: it
writes logs/control/<svc>.req (restart one), __restart__.req (restart all),
and logs/control/mode (normal|knowledge); the supervisor consumes them.

The monitor and watchdog tmux windows are folded in: the main loop processes
restart requests every second and, every VINKONA_WATCH_INTERVAL seconds,
checks each VINKONA_WATCH entry (name:port:rss_cap_MB) — reviving a dead LM
and pre-empting the llama.cpp embedding server's slow leak, exactly as before.

Placement rules are unchanged: on Linux+NVIDIA setups the Python services run
inside the distrobox container (VINKONA_BOX, default vinkona-cuda) and the
llama.cpp services on the host; without a usable container everything runs on
the host.  The knowledge host (Vinur) is just another service here when the
root vinkona.sh passes --kb <dir> — one supervisor, one status, no tmux.

Windows: process-group control below is POSIX; the Windows branch lands with
the platform port (job objects / taskkill).  Everything else is portable.

Usage (normally via ./vinkona.sh, which is a thin shim over this):
  supervisor.py start [normal|knowledge] [--kb DIR] [--kb-only]
  supervisor.py stop | restart [svc|normal|knowledge] | status | plan | mode
  supervisor.py logs [svc]      # follow one log, or all multiplexed
"""

# This file is the bootstrap: vinkona.sh runs it with the SYSTEM python3,
# before any env exists.  It must stay stdlib-only and 3.9-compatible — the
# macOS system python3 is 3.9, where `int | None` annotations evaluate eagerly
# (test_supervisor_compat.py gates this).
from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

if sys.version_info < (3, 9):
    sys.exit("vinkona supervisor needs Python 3.9+ (this python3 is %d.%d)"
             % sys.version_info[:2])

DIR = Path(__file__).resolve().parent
LOGS = DIR / "logs"
CTRL = LOGS / "control"
PIDFILE = CTRL / "supervisor.pid"
STATE = CTRL / "supervisor.state.json"     # child pids — read by status / stale cleanup
TOPO = CTRL / "topology.json"              # kb_dir / kb_only, persisted across restarts
MODE_FILE = CTRL / "mode"

BOX = os.environ.get("VINKONA_BOX", "vinkona-cuda")
WATCH_INTERVAL = int(os.environ.get("VINKONA_WATCH_INTERVAL", "20") or 20)
WATCHDOG_ON = os.environ.get("VINKONA_WATCHDOG", "1") != "0"
GRACE_S = 8                                # SIGTERM -> SIGKILL grace, as before


def _size_mb(s) -> int:
    """systemd-style size ('8G', '6144M', bytes) -> MB; 0 if unset/invalid."""
    try:
        s = str(s or "").strip().upper()
        if not s:
            return 0
        mult = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}.get(s[-1])
        if mult is None:
            return int(int(float(s)) / (1024 * 1024))
        return int(float(s[:-1]) * mult)
    except (ValueError, IndexError):
        return 0


def watch_specs() -> str:
    """VINKONA_WATCH, or a default derived from config: soft-restart the embed
    LM at 75% of its hard cgroup cap (embed_lm.mem_max), so the graceful
    watchdog restart usually beats the kernel's OOM kill — one knob (mem_max)
    scales both.  6000 MB when no cap is configured.  When the embed recycler
    is on (llm_server.py's default), the leaky llama-server child sits on
    port+10000 and the public port belongs to the small proxy — probe the
    child, it's the process whose RSS matters."""
    env = os.environ.get("VINKONA_WATCH")
    if env is not None:
        return env
    lm = load_config().get("embed_lm") or {}
    try:
        port = int(str(lm.get("url", "")).rsplit(":", 1)[-1].strip("/"))
    except ValueError:
        port = 11437
    if lm.get("recycle_rss_mb", None) not in (0, "0", False):   # absent → recycler on
        port = port + 10000 if port + 10000 <= 65535 else port - 10000
    cap = _size_mb(lm.get("mem_max"))
    soft = int(cap * 0.75) if cap else 6000
    return f"embed:{port}:{soft}"


# ── small helpers ─────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[supervisor] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def read_mode() -> str:
    try:
        m = MODE_FILE.read_text().strip()
    except OSError:
        m = ""
    return m if m in ("normal", "knowledge") else "normal"


def load_config() -> dict:
    try:
        return json.load(open(DIR / "config" / "config.json"))
    except Exception:
        return {}


def tts_engine(cfg: dict) -> str:
    # Legacy "orpheus" (the retired vLLM engine) maps to orpheus_gguf.
    eng = (cfg.get("tts") or {}).get("engine") or "orpheus_gguf"
    return eng if eng in ("neutts", "chatterbox") else "orpheus_gguf"


def load_topo() -> dict:
    try:
        return json.load(open(TOPO))
    except Exception:
        return {}


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def box_ok() -> bool:
    """Is the distrobox container usable?  Cached for the process lifetime."""
    if not hasattr(box_ok, "_v"):
        ok = False
        try:
            out = subprocess.run(["distrobox", "list"], capture_output=True,
                                 text=True, timeout=15).stdout
            ok = any(BOX in line.split() or f"{BOX} " in line or f" {BOX}" in line
                     for line in out.splitlines()) or BOX in out
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        box_ok._v = ok
    return box_ok._v


# ── service topology (mirrors the old set_services) ─────────────────────────

def services_for(mode: str, topo: dict) -> list[dict]:
    """Each entry: name, where (host|box|kb), cmd (argv), killpat (box reaping)."""
    kb_dir, kb_only = topo.get("kb_dir"), topo.get("kb_only")
    svcs: list[dict] = []

    def add(name, where, cmd, killpat=""):
        svcs.append({"name": name, "where": where, "cmd": cmd, "killpat": killpat})

    if not kb_only:
        if mode == "knowledge":
            add("big_lm", "host", ["./serve_big_lm.sh"])
            add("big_lm2", "host", ["./serve_big_lm2.sh"])
            add("embed", "host", ["./serve_embed.sh"])
            add("config", "box", ["./serve_config.sh"], r"config_server\.py")
        else:
            add("fast_lm", "host", ["./serve_fast_lm.sh"])
            add("big_lm", "host", ["./serve_big_lm.sh"])
            add("embed", "host", ["./serve_embed.sh"])
            add("tunnel", "host", ["./serve_tunnel.sh"])
            eng = tts_engine(load_config())
            if eng == "orpheus_gguf":          # only Orpheus needs the tts_lm llama-server
                add("tts_lm", "host", ["./serve_tts_lm.sh"])
            add("tts", "box", ["./serve_tts.sh", eng], r"tts_server\.py")
            add("cascade", "box", ["./serve_cascade.sh"], r"cascade_server\.py")
            add("config", "box", ["./serve_config.sh"], r"config_server\.py")
            add("research", "box", ["./serve_research.sh"], r"research_worker\.py")
    if kb_dir:
        add("kb", "kb", ["./run.sh"])
    return svcs


# ── process control (POSIX; the Windows branch lands with the platform port) ─

def spawn(svc: dict) -> subprocess.Popen:
    """Launch one service in its own process group, stdout -> logs/<name>.log."""
    logf = open(LOGS / f"{svc['name']}.log", "wb")     # truncate, like the old tee
    if svc["where"] == "box" and box_ok():
        inner = f"cd {shlex.quote(str(DIR))} && {' '.join(shlex.quote(a) for a in svc['cmd'])}"
        argv, cwd = ["distrobox", "enter", BOX, "--", "bash", "-lc", inner], str(DIR)
    elif svc["where"] == "kb":
        argv, cwd = svc["cmd"], load_topo().get("kb_dir")
    else:                                              # host, or box degraded to host
        argv, cwd = svc["cmd"], str(DIR)
    return subprocess.Popen(argv, cwd=cwd, stdout=logf, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def kill_group(pid: int) -> None:
    """TERM the process group, wait up to GRACE_S, then KILL stragglers."""
    try:
        pgid = os.getpgid(pid)
    except (OSError, ProcessLookupError):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return
    for _ in range(GRACE_S):
        if not pid_alive(pid):
            return
        time.sleep(1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


def bracket(pat: str) -> str:
    """'tts_server\\.py|x' -> '[t]ts_server\\.py|[x]' — the classic pgrep trick
    so a reaper's own cmdline can never match the pattern it carries."""
    parts = pat.split("|")
    return "|".join(f"[{p[0]}]{p[1:]}" if p else p for p in parts)


def reap_box_pattern(pat: str) -> None:
    """Kill matching processes INSIDE the container (detached workers included),
    so a restart can't collide with an orphan still holding VRAM."""
    if not pat:
        return
    safe = bracket(pat)
    script = (f'pkill -TERM -f "$VK_REAP_PAT" 2>/dev/null; '
              f'for i in 1 2 3 4 5 6 7 8; do pgrep -f "$VK_REAP_PAT" >/dev/null 2>&1 || break; sleep 1; done; '
              f'pkill -KILL -f "$VK_REAP_PAT" 2>/dev/null; true')
    if box_ok():
        subprocess.run(["distrobox", "enter", BOX, "--", "env", f"VK_REAP_PAT={safe}",
                        "bash", "-lc", script],
                       capture_output=True, timeout=60)
    else:
        env = dict(os.environ, VK_REAP_PAT=safe)
        subprocess.run(["bash", "-c", script], env=env, capture_output=True, timeout=60)


def reap_kb_orphans() -> None:
    """Knowledge-host ops jobs run in their own sessions (so the server can
    group-kill them) and survive a SIGKILLed parent — janitor them here.
    ('[-]m' is the same self-match guard as the old root vinkona.sh.)"""
    pat = "[-]m knowledgehost"
    if subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode != 0:
        return
    log("kb: reaping leftover worker processes")
    subprocess.run(["pkill", "-TERM", "-f", pat], capture_output=True)
    for _ in range(6):
        if subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode != 0:
            return
        time.sleep(1)
    subprocess.run(["pkill", "-KILL", "-f", pat], capture_output=True)


def rss_mb_for_port(port: int) -> int:
    """Total resident MB of processes whose cmdline carries '--port <port>'
    (the llama-server for that tier — same probe as the old watchdog)."""
    r = subprocess.run(["pgrep", "-f", f"--port {port}"], capture_output=True, text=True)
    total_kb = 0
    for pid in r.stdout.split():
        p = subprocess.run(["ps", "-o", "rss=", "-p", pid], capture_output=True, text=True)
        try:
            total_kb += int(p.stdout.strip() or 0)
        except ValueError:
            pass
    return total_kb // 1024


# ── the supervisor process ────────────────────────────────────────────────────

class Supervisor:
    def __init__(self):
        self.children: dict[str, subprocess.Popen] = {}
        self.svcs: list[dict] = []
        self.stopping = False
        self.watch = watch_specs()             # re-derived on each start_all (config can change)
        self.wd_last: dict[str, float] = {}    # name -> last watchdog restart ts

    def save_state(self):
        ordered = [s["name"] for s in self.svcs if s["name"] in self.children]
        state = {"pid": os.getpid(), "mode": read_mode(), "box": BOX,
                 "box_ok": box_ok(),
                 "services": {n: {"pid": self.children[n].pid,
                                  "where": self._svc(n)["where"],
                                  "exited": self.children[n].poll()}
                              for n in ordered}}
        STATE.write_text(json.dumps(state, indent=2))

    def _svc(self, name: str) -> dict:
        for s in self.svcs:
            if s["name"] == name:
                return s
        return {"name": name, "where": "host", "cmd": [], "killpat": ""}

    def start_all(self):
        self.svcs = services_for(read_mode(), load_topo())
        self.watch = watch_specs()
        # Wake the container ONCE, alone: racing four simultaneous `distrobox
        # enter` calls against a stopped container loses one of them.
        if any(s["where"] == "box" for s in self.svcs):
            if box_ok():
                log(f"waking the container ({BOX}) ...")
                subprocess.run(["distrobox", "enter", BOX, "--", "true"],
                               capture_output=True, timeout=120)
            else:
                log(f"no container '{BOX}' — running every service on the host")
        for s in self.svcs:
            try:
                self.children[s["name"]] = spawn(s)
                log(f"started {s['name']} (pid {self.children[s['name']].pid}, {s['where']})")
            except OSError as e:
                log(f"FAILED to start {s['name']}: {e}")
            if s["where"] == "host":
                time.sleep(1)                  # let the LM servers stagger their load
        self.save_state()

    def stop_one(self, name: str):
        svc = self._svc(name)
        child = self.children.pop(name, None)
        if child is not None and child.poll() is None:
            kill_group(child.pid)
        if svc["where"] == "box":
            reap_box_pattern(svc["killpat"])   # orphans holding VRAM
        if svc["where"] == "kb":
            reap_kb_orphans()

    def stop_all(self):
        """TERM every process group at once, wait one grace period, KILL the
        stragglers — parallel, so a full stop takes ~GRACE_S, not services×GRACE_S."""
        live = {n: c for n, c in self.children.items() if c.poll() is None}
        for c in live.values():
            try:
                os.killpg(os.getpgid(c.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
        deadline = time.time() + GRACE_S
        while time.time() < deadline and any(pid_alive(c.pid) for c in live.values()):
            time.sleep(0.5)
        for c in live.values():
            if pid_alive(c.pid):
                try:
                    os.killpg(os.getpgid(c.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass
        pats = [s["killpat"] for s in self.svcs
                if s["where"] == "box" and s["killpat"] and s["name"] in self.children]
        if pats:
            reap_box_pattern("|".join(pats))
        if any(s["where"] == "kb" and s["name"] in self.children for s in self.svcs):
            reap_kb_orphans()
        self.children.clear()
        self.save_state()

    def restart_one(self, name: str):
        if not any(s["name"] == name for s in self.svcs):
            log(f"restart request for unknown service: {name} "
                f"(known: {', '.join(s['name'] for s in self.svcs)})")
            return
        log(f"restarting {name}")
        self.stop_one(name)
        time.sleep(1)
        try:
            self.children[name] = spawn(self._svc(name))
        except OSError as e:
            log(f"FAILED to restart {name}: {e}")
        self.save_state()

    def full_restart(self):
        log(f"full restart (mode -> {read_mode()})")
        self.stop_all()
        time.sleep(2)
        self.start_all()

    def process_requests(self):
        for f in sorted(CTRL.glob("*.req")):
            svc = f.stem
            try:
                f.unlink()
            except OSError:
                continue
            if svc == "__restart__":
                self.full_restart()
            else:
                self.restart_one(svc)

    def watchdog_tick(self):
        """Revive dead watched LMs; pre-empt the embed RSS leak (VINKONA_WATCH)."""
        for spec in self.watch.split():
            try:
                name, port, cap = spec.split(":")
                port, cap = int(port), int(cap)
            except ValueError:
                continue
            if not any(s["name"] == name for s in self.svcs):
                continue
            now = time.time()
            if now - self.wd_last.get(name, 0) < WATCH_INTERVAL * 4:
                continue                       # cooldown: don't re-request mid-restart
            child = self.children.get(name)
            if child is None or child.poll() is not None:
                self.wd_last[name] = now
                log(f"watchdog: {name} not running -> restart")
                self.restart_one(name)
                continue
            if cap > 0:
                rss = rss_mb_for_port(port)
                if rss > cap:
                    self.wd_last[name] = now
                    log(f"watchdog: {name} RSS {rss}MB > {cap}MB -> restart")
                    self.restart_one(name)

    def run(self):
        CTRL.mkdir(parents=True, exist_ok=True)
        PIDFILE.write_text(str(os.getpid()))
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(self, "stopping", True))
        log(f"supervisor up (pid {os.getpid()}, mode {read_mode()}, box "
            f"{'ok' if box_ok() else 'absent -> host'})")
        self.start_all()
        last_wd = 0.0
        last_state = 0.0
        try:
            while not self.stopping:
                self.process_requests()
                if WATCHDOG_ON and time.time() - last_wd >= WATCH_INTERVAL:
                    last_wd = time.time()
                    self.watchdog_tick()
                if time.time() - last_state >= 10:     # keep exit codes fresh for status
                    last_state = time.time()
                    self.save_state()
                time.sleep(1)
        finally:
            log("stopping all services")
            self.stop_all()
            for p in (PIDFILE,):
                try:
                    p.unlink()
                except OSError:
                    pass
            log("stopped.")


# ── CLI verbs ─────────────────────────────────────────────────────────────────

def supervisor_pid() -> int | None:
    try:
        pid = int(PIDFILE.read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if pid_alive(pid) else None


def stale_cleanup():
    """A previous supervisor died without cleanup — kill what its state file
    still points at (exact pids, not patterns), then the box orphans."""
    try:
        state = json.load(open(STATE))
    except Exception:
        return
    killed = []
    for name, info in (state.get("services") or {}).items():
        pid = info.get("pid")
        if pid and info.get("exited") is None and pid_alive(pid):
            kill_group(pid)
            killed.append(name)
    if killed:
        print(f"cleaned up a stale run (killed: {', '.join(killed)})")


def missing_models():
    """[(tier, resolved_path)] for enabled LM tiers whose GGUF is absent — the
    start preflight.  Resolution goes through config.py's MERGED view (what
    llm_server.py actually uses: defaults + the user overlay, big_lm2
    inheriting big_lm), so the listed path is exactly where the service will
    look — a wrong models_dir shows itself here as an absolute path."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("vinkona_config",
                                                      str(DIR / "config.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = mod.load_config()
    except Exception as e:
        print(f"(model preflight skipped: {e})", file=sys.stderr)
        return []
    models_dir = Path(str(cfg.get("models_dir", "Models")))
    if not models_dir.is_absolute():
        models_dir = DIR / models_dir
    tiers = ["fast_lm", "big_lm", "big_lm2", "embed_lm"]
    if tts_engine(cfg) == "orpheus_gguf":
        tiers.append("tts_lm")
    out = []
    for tier in tiers:
        block = dict(cfg.get(tier) or {})
        if tier == "big_lm2":
            block = {**dict(cfg.get("big_lm") or {}), **block}
        if not block.get("url") or not block.get("model"):
            continue
        model = Path(str(block["model"]))
        path = model if model.is_absolute() else models_dir / model
        if not path.exists():
            out.append((tier, str(path)))
    return out


def preflight_models() -> None:
    miss = missing_models()
    if not miss:
        return
    print("model files missing for enabled tiers (services will refuse to serve):")
    for tier, p in miss:
        print(f"  {tier:<9} {p}")
    if sys.stdin.isatty():
        try:
            ans = input("fetch the default set now (fetch_models.sh, RAM-sized)? "
                        "[Y/n] ").strip().lower()
        except EOFError:
            ans = "n"
        if ans not in ("n", "no"):
            subprocess.run(["bash", str(DIR / "fetch_models.sh")], cwd=str(DIR))
            for tier, p in missing_models():
                print(f"  still missing: {tier}  {p}  — the config names a "
                      f"model the fetch set doesn't include; fix the tier's "
                      f"'model' (config web UI) or fetch it manually")
    else:
        print("  → ./vinkona.sh models   (or: cd assistant && ./fetch_models.sh)")


def cmd_start(args: list[str]) -> int:
    mode = None
    topo = load_topo()
    it = iter(args)
    for a in it:
        if a in ("normal", "knowledge"):
            mode = a
        elif a == "--kb":
            topo["kb_dir"] = str(Path(next(it)).resolve())
        elif a == "--kb-only":
            topo["kb_only"] = True
        elif a == "--assistant-only":
            topo.pop("kb_dir", None); topo["kb_only"] = False
        else:
            print(f"unknown start argument: {a}", file=sys.stderr); return 1
    CTRL.mkdir(parents=True, exist_ok=True)
    if mode:
        MODE_FILE.write_text(mode + "\n")
    TOPO.write_text(json.dumps(topo))
    if supervisor_pid():
        print("supervisor is already running — use './vinkona.sh restart' or 'status'.")
        return 0
    stale_cleanup()
    preflight_models()
    if os.name != "posix":
        print("the supervisor's process control is POSIX-only for now "
              "(Windows lands with the platform port)", file=sys.stderr)
        return 1
    logf = open(LOGS / "supervisor.log", "wb")
    p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_run"],
                         cwd=str(DIR), stdout=logf, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    for _ in range(20):                        # confirm liftoff
        if supervisor_pid():
            break
        if p.poll() is not None:
            print("supervisor failed to start — see logs/supervisor.log", file=sys.stderr)
            return 1
        time.sleep(0.25)
    svcs = services_for(read_mode(), topo)
    print(f"started the supervisor (pid {p.pid}, mode {read_mode()}, "
          f"{len(svcs)} services).  './vinkona.sh status' to check, "
          f"'./vinkona.sh logs' to watch.")
    return 0


def cmd_stop() -> int:
    pid = supervisor_pid()
    if pid:
        os.kill(pid, signal.SIGTERM)
        for _ in range(45):                    # parallel child shutdown + box reaps
            if not pid_alive(pid):
                print("stopped.")
                return 0
            time.sleep(1)
        print("supervisor didn't exit — killing it and its services")
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    stale_cleanup()
    # Janitor passes that exist regardless of who started things:
    if (load_topo().get("kb_dir")):
        reap_kb_orphans()
    print("stopped." if pid else "supervisor was not running (leftovers cleaned if any).")
    return 0


def cmd_restart(args: list[str]) -> int:
    pid = supervisor_pid()
    if args and args[0] in ("normal", "knowledge"):
        CTRL.mkdir(parents=True, exist_ok=True)
        MODE_FILE.write_text(args[0] + "\n")
        args = ["__restart__"]
    if not pid:
        print("supervisor not running — starting fresh")
        return cmd_start([])
    CTRL.mkdir(parents=True, exist_ok=True)
    target = args[0] if args else "__restart__"
    (CTRL / f"{target}.req").write_text(str(time.time()))
    print(f"requested restart: {'everything' if target == '__restart__' else target}")
    return 0


def _http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2):
            return True
    except Exception:
        return False


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            return True
    except OSError:
        return False


def _kb_port(kb_dir: str) -> int:
    try:
        for line in open(Path(kb_dir) / "config.toml"):
            line = line.split("#", 1)[0].strip()
            if line.startswith("port"):
                return int(line.split("=", 1)[1].strip().strip('"'))
    except (OSError, ValueError, IndexError):
        pass
    return 8771


def svc_check(name: str, cfg: dict, topo: dict, child_alive: bool) -> str:
    def tier_port(tier, default):
        return int((cfg.get(tier) or {}).get("port") or default)
    health = {"fast_lm": 11435, "big_lm": 11438, "big_lm2": 11440,
              "embed": 11437, "tts_lm": 11439}
    if name in health:
        port = tier_port(name, health[name])
        return (f"up            (:{port} — health ok)" if _http_ok(f"http://127.0.0.1:{port}/health")
                else "not answering" + (f" (:{port})" if child_alive else f" (:{port}, process exited)"))
    if name == "tts":
        port = int((cfg.get("tts") or {}).get("port") or 11436)
        return f"up            (:{port})" if _port_open(port) else "not listening"
    if name == "cascade":
        port = int((cfg.get("server") or {}).get("port") or 8998)
        return f"up            (:{port})" if _port_open(port) else "not listening"
    if name == "config":
        port = int((cfg.get("config_server") or {}).get("port") or 8090)
        return f"up            (:{port})" if _port_open(port) else "not listening"
    if name == "kb":
        port = _kb_port(topo.get("kb_dir", ""))
        return (f"up            (:{port} — health ok)" if _http_ok(f"http://127.0.0.1:{port}/health")
                else "not answering" + ("" if child_alive else " (process exited)"))
    if name == "tunnel":
        return ("up            (process)" if child_alive
                else "not running   (fine if tools.tunnel is off)")
    return "up            (process)" if child_alive else "not running"


def _last_log_line(name: str) -> str:
    """The most reason-looking recent line of logs/<name>.log — shown under a
    non-up service so 'not answering' carries its own diagnosis (the LM
    launcher's preflights exit with exactly these one-liners)."""
    try:
        data = (LOGS / f"{name}.log").read_bytes()[-4096:].decode("utf-8", "replace")
    except OSError:
        return ""
    lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
    for ln in reversed(lines):
        low = ln.lower()
        if any(k in low for k in ("error", "not found", "missing", "failed",
                                  "died", "exited", "no such", "traceback",
                                  "killed", "oom")):
            return ln[:140]
    return lines[-1][:140] if lines else ""


def status_payload() -> dict:
    """Everything cmd_status prints, machine-readable — the seam the desktop
    launcher polls (launcher/ Tauri app).  Always includes the web-UI URLs so
    the launcher never hardcodes ports; service entries appear only while the
    supervisor runs."""
    topo, cfg = load_topo(), load_config()
    kb_dir = topo.get("kb_dir") or ""
    payload = {
        "running": False, "supervisor": None, "mode": read_mode(),
        "services": [],
        "ui": {"config": "http://127.0.0.1:%d" % int(
                   (cfg.get("config_server") or {}).get("port") or 8090),
               "kb": ("http://127.0.0.1:%d" % _kb_port(kb_dir)) if kb_dir else None},
    }
    pid = supervisor_pid()
    if not pid:
        return payload
    payload["running"], payload["supervisor"] = True, pid
    try:
        state = json.load(open(STATE))
    except Exception:
        state = {}
    payload["mode"] = state.get("mode", payload["mode"])
    payload["box"] = BOX if state.get("box_ok") else None
    for name, info in (state.get("services") or {}).items():
        alive = info.get("exited") is None and pid_alive(info.get("pid", -1))
        line = svc_check(name, cfg, topo, alive)
        item = {"name": name, "pid": info.get("pid"),
                "up": line.startswith("up"), "detail": " ".join(line.split())}
        if not item["up"] and "fine if" not in line:
            hint = _last_log_line(name)
            if hint:
                item["reason"] = hint
        payload["services"].append(item)
    return payload


def cmd_status_json() -> int:
    print(json.dumps(status_payload()))
    return 0


def cmd_status() -> int:
    pid = supervisor_pid()
    if not pid:
        print("supervisor not running")
        return 1
    try:
        state = json.load(open(STATE))
    except Exception:
        state = {}
    topo, cfg = load_topo(), load_config()
    boxinfo = f"box {BOX}" if state.get("box_ok") else "host-only"
    print(f"supervisor up (pid {pid}, mode: {state.get('mode', read_mode())}, {boxinfo})")
    for name, info in (state.get("services") or {}).items():
        alive = info.get("exited") is None and pid_alive(info.get("pid", -1))
        line = svc_check(name, cfg, topo, alive)
        print(f"  {name:<9} {line}")
        if not line.startswith("up") and "fine if" not in line:
            hint = _last_log_line(name)
            if hint:
                print(f"  {'':<9} ↳ {hint}")
    return 0


def cmd_plan() -> int:
    topo = load_topo()
    for s in services_for(read_mode(), topo):
        where = s["where"]
        if where == "box" and not box_ok():
            where = "box->host"
        cwd = topo.get("kb_dir") if s["where"] == "kb" else str(DIR)
        print(f"{s['name']:<9} [{where}]  cd {cwd} && {' '.join(s['cmd'])}"
              f"  > logs/{s['name']}.log")
    return 0


def cmd_logs(args: list[str]) -> int:
    """Follow one service log, or all of them multiplexed (name | line)."""
    if args:
        files = [LOGS / f"{args[0]}.log"]
        if not files[0].exists():
            print(f"no log yet: {files[0]}", file=sys.stderr)
            return 1
    else:
        files = sorted(LOGS.glob("*.log"))
        if not files:
            print("no logs yet — start the stack first", file=sys.stderr)
            return 1
    handles = {}
    for f in files:
        try:
            h = open(f, "rb")
            h.seek(0, 2)                       # follow from the end
            handles[f.stem] = h
        except OSError:
            pass
    width = max(len(n) for n in handles) if handles else 0
    print(f"following {', '.join(sorted(handles))}  (Ctrl-C to stop)", flush=True)
    try:
        while True:
            wrote = False
            for name, h in handles.items():
                for line in h.read().splitlines():
                    text = line.decode("utf-8", "replace")
                    print(f"{name:<{width}} | {text}" if len(handles) > 1 else text,
                          flush=True)
                    wrote = True
            if not wrote:
                time.sleep(0.5)
    except KeyboardInterrupt:
        return 0


def main() -> int:
    LOGS.mkdir(exist_ok=True)
    CTRL.mkdir(parents=True, exist_ok=True)
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if cmd == "start":
        return cmd_start(args)
    if cmd == "stop":
        return cmd_stop()
    if cmd == "restart":
        return cmd_restart(args)
    if cmd == "status":
        return cmd_status_json() if "--json" in args else cmd_status()
    if cmd == "plan":
        return cmd_plan()
    if cmd == "mode":
        print(read_mode())
        return 0
    if cmd == "logs":
        return cmd_logs(args)
    if cmd == "_run":
        Supervisor().run()
        return 0
    print(__doc__.strip().split("Usage", 1)[-1] if "Usage" in (__doc__ or "") else
          "usage: supervisor.py {start|stop|restart|status|plan|mode|logs}",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
