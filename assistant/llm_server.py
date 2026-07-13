#!/usr/bin/env python
"""
LM launcher — start one llama.cpp `llama-server` for a config tier.

Each language-model tier (fast_lm / big_lm / embed_lm / tts_lm) in config/config.json
says which GGUF to load, which GPU to pin it to, and the llama.cpp knobs.  This turns
a tier block into a `llama-server` command, pins it to the chosen GPU, and execs it.

  python llm_server.py --tier fast_lm     # or big_lm / embed_lm / tts_lm
  python llm_server.py --tier big_lm --dry-run    # print the command, don't run

GPU pinning uses CUDA_DEVICE_ORDER=PCI_BUS_ID so the `gpu` index matches the other
services (TTS, cascade).  CUDA_VISIBLE_DEVICES is set to that single device, so
llama-server sees exactly one GPU and `-ngl` puts the model there.

The binary is config `llama_bin` (default `llama-server` on PATH); the LLAMA_SERVER
env var overrides it.  Models resolve under config `models_dir` unless absolute.

The embed tier is special: llama.cpp's embedding server grows its RSS without
bound under sustained batch load, and there is no in-process scrub — only
process exit returns the memory.  So for that tier llm_server.py does not exec:
it stays resident as a small stdlib reverse proxy (the "recycler"), runs the
real llama-server as a child on 127.0.0.1:<port+10000>, and restarts the child
BETWEEN requests once its RSS crosses embed_lm.recycle_rss_mb.  Clients never
see a dropped connection — a request that lands mid-recycle just waits the few
seconds the model takes to reload.  Set recycle_rss_mb to 0 to disable and get
the plain exec behaviour back.
"""

import argparse
import http.client
import http.server
import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path


def _load_cfgmod():
    spec = importlib.util.spec_from_file_location("config", str(Path(__file__).parent / "config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_command(cfg: dict, tier: str) -> tuple[list[str], dict, Path]:
    """Return (argv, env, model_path) for launching `tier`'s llama-server."""
    cfgmod = _load_cfgmod()
    if tier not in cfgmod.LM_TIERS:
        sys.exit(f"unknown tier '{tier}'; choose one of {', '.join(cfgmod.LM_TIERS)}")
    block = cfg.get(tier) or {}
    if tier == "big_lm2":               # second big-LM instance: inherit big_lm, override per big_lm2
        block = {**(cfg.get("big_lm") or {}), **block}
    if not block.get("url"):
        sys.exit(f"tier '{tier}' has no url set (it is disabled)")
    if not block.get("model"):
        sys.exit(f"tier '{tier}' has no model set")

    host, port = cfgmod.lm_bind(block["url"])
    models_dir = Path(cfg.get("models_dir", "Models"))
    model = Path(block["model"])
    model_path = model if model.is_absolute() else models_dir / model

    binary = os.environ.get("LLAMA_SERVER") or cfg.get("llama_bin", "llama-server")
    cmd = [binary,
           "-m", str(model_path),
           "--host", host, "--port", str(port),
           "-c", str(block.get("ctx_size", 4096)),
           "-ngl", str(block.get("n_gpu_layers", 99))]
    # Flash attention: recent llama-server wants a value (--flash-attn on|off|auto);
    # accept a bool (True→on, False→omit) or an explicit string from config.
    fa = block.get("flash_attn", False)
    if isinstance(fa, str):
        cmd += ["--flash-attn", fa]
    elif fa:
        cmd += ["--flash-attn", "on"]
    if tier == "embed_lm" or block.get("embedding"):
        cmd += ["--embedding", "--pooling", block.get("pooling", "mean")]
    # --jinja makes llama-server use the model's own chat template, which is what
    # renders tool_calls / tool-result messages back into the prompt.  Without it the
    # fast LM can emit a tool call but never sees the tool's result on the follow-up
    # turn ("let me check" → silence).  Needed on any tier that uses Tier-2 tools.
    if block.get("jinja"):
        cmd += ["--jinja"]
    # Memory levers (a big model near the VRAM ceiling OOMs at graph capture):
    #  • parallel: serving slots — this is a single-user cascade, so 1 is right and
    #    avoids llama.cpp's auto n_parallel=4 reserving extra KV/compute buffers.
    #  • cache_type_k/v: quantise the KV cache (e.g. "q8_0") to roughly halve its
    #    footprint; needs flash-attn on (it is, by default).
    if block.get("parallel"):
        cmd += ["--parallel", str(block["parallel"])]
    if block.get("cache_type_k"):
        cmd += ["-ctk", str(block["cache_type_k"])]
    if block.get("cache_type_v"):
        cmd += ["-ctv", str(block["cache_type_v"])]
    cmd += [str(a) for a in block.get("extra_args", [])]

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(block.get("gpu", 0))
    return cmd, env, model_path


# ── embed recycler ────────────────────────────────────────────────────────────

def _size_mb(s) -> int:
    """systemd-style size ('8G', '6144M', bytes) -> MB; 0 if unset/invalid.
    (supervisor.py carries its own copy — it deliberately imports nothing.)"""
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


def recycle_threshold_mb(block: dict) -> int:
    """MB of child RSS that triggers a graceful restart; 0 = recycler off.
    Default derives from mem_max (half of it) so mem_max stays the one knob:
    recycle at 50%, watchdog TERM at 75%, cgroup kill at 100%."""
    v = block.get("recycle_rss_mb", None)
    if v in (0, "0", False):
        return 0
    try:
        if v is not None and str(v).strip():
            return max(int(float(v)), 0)
    except (TypeError, ValueError):
        pass
    cap = _size_mb(block.get("mem_max"))
    return cap // 2 if cap else 3000


def recycle_child_port(port: int) -> int:
    """Where the llama-server child hides when the recycler owns `port`.
    (supervisor.watch_specs mirrors this so the watchdog probes the child.)"""
    return port + 10000 if port + 10000 <= 65535 else port - 10000


class Recycler:
    """Reverse proxy that owns the public port and restarts its llama-server
    child between requests whenever the child's RSS crosses cap_mb."""

    def __init__(self, argv: list[str], env: dict, child_port: int, cap_mb: int):
        self.argv, self.env = argv, env
        self.child_port, self.cap_mb = child_port, cap_mb
        self.child: subprocess.Popen | None = None
        self.cond = threading.Condition()     # guards active/recycling
        self.active = 0                       # in-flight forwards
        self.recycling = False
        self.spawn_lock = threading.Lock()    # serializes crash respawns
        self.requests = 0
        self.recycles = 0

    # -- child lifecycle -------------------------------------------------------

    def start_child(self, wait_s: float = 300.0):
        # No new session: the child stays in our process group, so the
        # supervisor's group TERM/KILL takes both of us down together.
        self.child = subprocess.Popen(self.argv, env=self.env)
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if self.child.poll() is not None:
                raise RuntimeError(f"llama-server exited (code {self.child.returncode})")
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{self.child_port}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except OSError:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"llama-server not healthy within {wait_s:.0f}s")

    def stop_child(self):
        c = self.child
        if c is None or c.poll() is not None:
            return
        c.terminate()
        try:
            c.wait(timeout=10)
        except subprocess.TimeoutExpired:
            c.kill()
            try:
                c.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def child_rss_mb(self) -> int:
        if self.child is None:
            return 0
        pid = self.child.pid
        try:                                   # Linux: free and exact
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) // 1024
        except (OSError, ValueError, IndexError):
            pass
        try:                                   # macOS and friends
            out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                                 capture_output=True, text=True).stdout.strip()
            return int(out or 0) // 1024
        except (OSError, ValueError):
            return 0

    # -- request path ----------------------------------------------------------

    def forward(self, method: str, path: str, headers, body):
        """Proxy one request; returns (status, bytes, content_type)."""
        with self.cond:
            while self.recycling:
                self.cond.wait()
            self.active += 1
        try:
            for attempt in (1, 2):
                try:
                    return self._forward_once(method, path, headers, body)
                except (OSError, http.client.HTTPException):
                    if attempt == 2:
                        return (502, b'{"error":"embed backend unavailable"}',
                                "application/json")
                    self._respawn_if_dead()    # child crashed under us? revive, retry once
        finally:
            self._finish_and_maybe_recycle()

    def _forward_once(self, method, path, headers, body):
        conn = http.client.HTTPConnection("127.0.0.1", self.child_port, timeout=600)
        try:
            fwd = {k: v for k, v in headers.items()
                   if k.lower() in ("content-type", "authorization", "accept")}
            conn.request(method, path, body=body, headers=fwd)
            resp = conn.getresponse()
            return (resp.status, resp.read(),
                    resp.getheader("Content-Type") or "application/json")
        finally:
            conn.close()

    def _respawn_if_dead(self):
        with self.spawn_lock:
            if self.child is not None and self.child.poll() is None:
                return
            print("[recycle] llama-server died — respawning", flush=True)
            try:
                self.start_child()
            except (OSError, RuntimeError) as e:
                print(f"[recycle] respawn failed: {e}", flush=True)

    def _finish_and_maybe_recycle(self):
        mine = False
        with self.cond:
            self.active -= 1
            self.requests += 1
            if (not self.recycling and self.cap_mb
                    and self.child_rss_mb() >= self.cap_mb):
                self.recycling = mine = True   # this thread owns the recycle
            self.cond.notify_all()
        if mine:
            self._recycle()

    def _recycle(self):
        with self.cond:
            while self.active > 0:             # drain in-flight requests
                self.cond.wait()
        rss, t0 = self.child_rss_mb(), time.time()
        print(f"[recycle] llama-server RSS {rss}MB >= {self.cap_mb}MB after "
              f"{self.requests} requests — restarting it between requests", flush=True)
        try:
            self.stop_child()
            self.start_child()
            self.recycles += 1
            print(f"[recycle] fresh llama-server up in {time.time() - t0:.1f}s "
                  f"(recycle #{self.recycles})", flush=True)
        except (OSError, RuntimeError) as e:
            print(f"[recycle] RESTART FAILED: {e} — retrying on the next request",
                  flush=True)
        finally:
            with self.cond:
                self.recycling = False
                self.cond.notify_all()


def _make_handler(recycler: Recycler):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):          # the llama child logs plenty
            pass

        def _serve(self):
            if self.path == "/health" and recycler.recycling:
                self._reply(503, b'{"status":"recycling"}', "application/json")
                return
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else None
            self._reply(*recycler.forward(self.command, self.path, self.headers, body))

        def _reply(self, status, data, ctype):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = _serve

    return Handler


def run_recycler(cmd: list[str], env: dict, cap_mb: int) -> None:
    host = cmd[cmd.index("--host") + 1]
    port = int(cmd[cmd.index("--port") + 1])
    child_port = recycle_child_port(port)
    argv = list(cmd)
    argv[argv.index("--host") + 1] = "127.0.0.1"     # child is loopback-only
    argv[argv.index("--port") + 1] = str(child_port)

    recycler = Recycler(argv, env, child_port, cap_mb)

    def _term(*_):
        recycler.stop_child()
        os._exit(0)
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)

    print(f"[recycle] :{port} -> llama-server :{child_port}; graceful restart at "
          f"{cap_mb}MB RSS (embed_lm.recycle_rss_mb; 0 disables)", flush=True)
    recycler.start_child()
    http.server.ThreadingHTTPServer((host, port), _make_handler(recycler)).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, help="fast_lm | big_lm | embed_lm | tts_lm")
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = ap.parse_args()

    cfg = _load_cfgmod().load_config(args.config)
    cmd, env, model_path = build_command(cfg, args.tier)
    block = cfg.get(args.tier) or {}
    recycle_mb = (recycle_threshold_mb(block)
                  if args.tier == "embed_lm" or block.get("embedding") else 0)

    print(f"[lm] {args.tier}: GPU {env['CUDA_VISIBLE_DEVICES']}  "
          f"{cmd[cmd.index('--host')+1]}:{cmd[cmd.index('--port')+1]}  {model_path}", flush=True)
    print("[lm] " + " ".join(cmd), flush=True)
    if args.dry_run:
        if recycle_mb:
            port = int(cmd[cmd.index("--port") + 1])
            print(f"[lm] embed recycler on: llama-server moves to "
                  f":{recycle_child_port(port)}, restart at {recycle_mb}MB RSS", flush=True)
        return

    binary = cmd[0]
    if shutil.which(binary) is None and not Path(binary).exists():
        sys.exit(f"[lm] llama-server binary '{binary}' not found — set llama_bin in "
                 f"config or the LLAMA_SERVER env var to your llama.cpp build.")
    if not model_path.exists():
        sys.exit(f"[lm] model not found: {model_path}\n"
                 f"     put the GGUF under {cfg.get('models_dir', 'Models')}/ "
                 f"(symlink it if it lives elsewhere), or run ./fetch_models.sh")

    if recycle_mb:
        run_recycler(cmd, env, recycle_mb)
        return
    os.execvpe(binary, cmd, env)


if __name__ == "__main__":
    main()
