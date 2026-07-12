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
"""

import argparse
import importlib.util
import os
import shutil
import sys
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, help="fast_lm | big_lm | embed_lm | tts_lm")
    ap.add_argument("--config", default="config/config.json")
    ap.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = ap.parse_args()

    cfg = _load_cfgmod().load_config(args.config)
    cmd, env, model_path = build_command(cfg, args.tier)

    print(f"[lm] {args.tier}: GPU {env['CUDA_VISIBLE_DEVICES']}  "
          f"{cmd[cmd.index('--host')+1]}:{cmd[cmd.index('--port')+1]}  {model_path}", flush=True)
    print("[lm] " + " ".join(cmd), flush=True)
    if args.dry_run:
        return

    binary = cmd[0]
    if shutil.which(binary) is None and not Path(binary).exists():
        sys.exit(f"[lm] llama-server binary '{binary}' not found — set llama_bin in "
                 f"config or the LLAMA_SERVER env var to your llama.cpp build.")
    if not model_path.exists():
        sys.exit(f"[lm] model not found: {model_path}\n"
                 f"     put the GGUF under {cfg.get('models_dir', 'Models')}/ "
                 f"(symlink it if it lives elsewhere), or run ./fetch_models.sh")

    os.execvpe(binary, cmd, env)


if __name__ == "__main__":
    main()
