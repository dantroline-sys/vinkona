#!/usr/bin/env python
"""
TTS HTTP service — wraps one TTS engine (Orpheus on llama.cpp, or NeuTTS) in
the right venv and exposes synthesis over HTTP, so the main server (a different
venv) can call it like the local LLM instances.  Stdlib-only HTTP (no extra deps
beyond the engine + soundfile), engine loaded once at startup so any warmup is
paid a single time.

Run INSIDE the engine's venv:

  # Orpheus on llama.cpp (vinkona_env) — preset voices + inline <laugh>/<sigh>
  # tags; needs the tts_lm llama-server running (serve_tts_lm.sh)
  python tts_server.py --engine orpheus_gguf

  # NeuTTS (neutts_env) — cloned voice from a reference clip
  CUDA_VISIBLE_DEVICES=0 python tts_server.py --engine neutts \
      --voice vinkona --ref voices/vinkona.wav --port 11436

Endpoints:
  GET  /health      -> {"status":"ok","engine":..,"sample_rate":..,"voices":[..]}
  GET  /voices      -> {"voices":[..]}
  POST /synthesize  {"text":"..","voice":".."} -> audio/wav (24 kHz, PCM_16)
"""

import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [tts] {msg}", flush=True)


def build_engine(engine: str, cfg: dict, device: str):
    """Lazy-import and configure the selected engine from config (each engine
    lives in its own venv, so we only import the one we're running)."""
    tts = cfg["tts"]
    if engine == "neutts":
        from tts_neutts import NeuTTSEngine
        nt = tts["neutts"]
        eng = NeuTTSEngine(backbone_repo=nt["backbone"], device=device)
        ref = nt.get("ref_wav")
        if not ref:
            raise SystemExit("config tts.neutts.ref_wav is required for the neutts engine")
        eng.register_voice(tts["default_voice"], ref, ref_text=nt.get("ref_text"))
        return eng
    if engine == "chatterbox":
        from tts_chatterbox import ChatterboxEngine
        cb = tts["chatterbox"]
        eng = ChatterboxEngine(device=device,
                               exaggeration=cb["exaggeration"],
                               cfg_weight=cb["cfg_weight"],
                               temperature=cb["temperature"])
        # ref_wav null = the built-in voice; a clip clones the persona voice.
        eng.register_voice(tts["default_voice"], cb.get("ref_wav"))
        return eng
    if engine == "orpheus_gguf":
        from tts_orpheus_gguf import OrpheusGGUFEngine
        og = tts["orpheus_gguf"]
        lm_url = og.get("lm_url") or (cfg.get("tts_lm") or {}).get("url") \
            or "http://127.0.0.1:11439"
        return OrpheusGGUFEngine(lm_url=lm_url,
                                 default_voice=tts["default_voice"],
                                 snac_path=og.get("snac_path"),
                                 snac_repo=og["snac_repo"],
                                 snac_file=og["snac_file"],
                                 temperature=og["temperature"],
                                 top_p=og["top_p"],
                                 repeat_penalty=og["repeat_penalty"],
                                 max_tokens=og["max_tokens"])
    raise SystemExit(f"unknown engine {engine}")


class Handler(BaseHTTPRequestHandler):
    engine = None            # set on the class before serving
    engine_name = None
    default_voice = None
    # Serialize synthesis — one GPU/engine, requests are per-turn anyway.
    synth_lock = threading.Lock()

    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok", "engine": self.engine_name,
                "sample_rate": self.engine.sample_rate, "voices": self.engine.voices,
            })
        elif self.path == "/voices":
            self._send_json(200, {"voices": self.engine.voices})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/synthesize", "/synthesize_stream"):
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:
            self._send_json(400, {"error": f"bad request: {e}"})
            return
        text = (req.get("text") or "").strip()
        voice = req.get("voice") or self.default_voice
        if not text:
            self._send_json(400, {"error": "empty text"})
            return

        if self.path == "/synthesize_stream":
            self._do_stream(text, voice)
            return

        t0 = time.monotonic()
        try:
            with self.synth_lock:
                pcm = self.engine.synthesize(text, voice=voice)
        except Exception as e:
            self._send_json(500, {"error": f"synthesis failed: {e}"})
            return
        dt = time.monotonic() - t0
        dur = len(pcm) / self.engine.sample_rate if len(pcm) else 0.0
        _log(f"synth voice={voice} {len(text)}ch -> {dur:.2f}s in {dt:.2f}s "
             f"(RTF {dt/dur:.2f})" if dur else f"synth produced no audio")

        buf = io.BytesIO()
        sf.write(buf, pcm, self.engine.sample_rate, format="WAV", subtype="PCM_16")
        data = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _do_stream(self, text: str, voice: str):
        # Stream raw 16-bit PCM @ 24 kHz as the engine produces it, so the client
        # can start playing before the whole utterance is synthesized.  HTTP/1.0
        # closes the connection at the end, which frames the stream for the client.
        self.send_response(200)
        self.send_header("Content-Type", "audio/L16;rate=24000;channels=1")
        self.end_headers()
        t0 = time.monotonic()
        n = 0
        try:
            with self.synth_lock:
                for chunk in self.engine.synthesize_stream(text, voice):
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    n += len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return                      # client hung up (barge-in) — fine
        except Exception as e:
            _log(f"stream synth failed: {e}")
            return
        dur = n / 2 / self.engine.sample_rate
        _log(f"stream voice={voice} {len(text)}ch -> {dur:.2f}s in {time.monotonic()-t0:.2f}s")

    def log_message(self, *args):  # silence the default per-request stderr noise
        pass


def main():
    import importlib.util
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--engine", choices=["neutts", "orpheus", "orpheus_gguf", "chatterbox"], default=None,
                    help="override config tts.engine (also selects which venv's deps to load)")
    ap.add_argument("--device", default="auto",
                    help="auto (cuda > mps > cpu), or an explicit torch device")
    args = ap.parse_args()

    # Load config.py from the repo root (works in any engine venv).
    _spec = importlib.util.spec_from_file_location("config", "config.py")
    _cfgmod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_cfgmod)
    cfg = _cfgmod.load_config(args.config)

    engine = args.engine or cfg["tts"]["engine"]
    host, port = cfg["tts"]["host"], cfg["tts"]["port"]

    _log(f"loading engine '{engine}' ...")
    t0 = time.monotonic()
    engine_obj = build_engine(engine, cfg, args.device)
    Handler.engine = engine_obj
    Handler.engine_name = engine
    Handler.default_voice = cfg["tts"]["default_voice"] or (engine_obj.voices[0] if engine_obj.voices else None)
    _log(f"engine ready in {time.monotonic()-t0:.1f}s  voices={engine_obj.voices}  "
         f"default={Handler.default_voice}")
    engine = engine_obj  # for warmup below

    # Warm up so the first real request doesn't pay the Triton JIT / first-call
    # compilation spike (that's the inflated RTF you'd otherwise see on turn 1).
    _log("warming up ...")
    try:
        t0 = time.monotonic()
        for _ in engine.synthesize_stream("Hello there.", Handler.default_voice):
            pass
        _log(f"warmed in {time.monotonic()-t0:.1f}s")
    except Exception as e:
        _log(f"warmup failed (non-fatal): {e}")

    server = ThreadingHTTPServer((host, port), Handler)
    _log(f"serving on http://{host}:{port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("shutting down")


if __name__ == "__main__":
    main()
