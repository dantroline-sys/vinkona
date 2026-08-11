#!/bin/bash
# Start the TTS service in the correct venv for the chosen engine.
#
#   ./serve_tts.sh orpheus_gguf # uses vinkona_env (llama.cpp backbone — needs
#                               #   the tts_lm llama-server, see serve_tts_lm.sh)
#   ./serve_tts.sh neutts       # uses neutts_env
#   ./serve_tts.sh chatterbox   # uses chatterbox_env (low-footprint, ~0.5B)
#
# All settings (port, voice, model, gpu mem, refs) come from config/config.json;
# the engine arg only selects which venv to activate.  Override the GPU with
# CUDA_VISIBLE_DEVICES=N ./serve_tts.sh ...
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"          # in-tree caches/tmp/PATH — see env.sh
ENGINE="${1:-orpheus_gguf}"
CONFIG="$SCRIPT_DIR/config/config.json"

# Stable GPU ordering so every process agrees on device indices.
# The live response path is fast LM + embed + TTS together on the 4090; the big LM
# runs alone on the 3090.  On this box the 4090 is CUDA index 1 (verify with
# CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi).  Override with
# CUDA_VISIBLE_DEVICES=N ./serve_tts.sh ...
export CUDA_DEVICE_ORDER=PCI_BUS_ID
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES

# Self-provision: if the chosen engine isn't installed yet, install it now
# (install.sh is idempotent).  THIS is what makes "switch TTS engine" in the config
# UI just work — the panel writes tts.engine and restarts; we build the right venv
# before serving.  The process stays alive throughout, so the supervisor (which only
# respawns on exit) doesn't mistake the install for a crash.
_tts_installed() {
  case "$1" in
    orpheus_gguf) ls "$SCRIPT_DIR"/vinkona_env/lib/python*/site-packages/onnxruntime >/dev/null 2>&1 ;;
    neutts)       ls "$SCRIPT_DIR"/neutts_env/lib/python*/site-packages/numpy >/dev/null 2>&1 ;;
    chatterbox)   ls -d "$SCRIPT_DIR"/chatterbox_env/lib/python*/site-packages/chatterbox >/dev/null 2>&1 ;;
    qwen3)        ls -d "$SCRIPT_DIR"/qwen3_env/lib/python*/site-packages/qwen_tts >/dev/null 2>&1 ;;
    *) return 0 ;;
  esac
}
if ! _tts_installed "$ENGINE"; then
  echo "[serve_tts] engine '$ENGINE' not installed yet — installing it now (one-time) ..." >&2
  (cd "$SCRIPT_DIR" && ./install.sh tts "$ENGINE") \
    || { echo "[serve_tts] install of '$ENGINE' failed — see above" >&2; exit 1; }
fi

case "$ENGINE" in
  orpheus_gguf) source "$SCRIPT_DIR/vinkona_env/bin/activate" ;;   # no engine venv: the
                                # backbone is the tts_lm llama-server, SNAC runs on CPU
  neutts)     source "$SCRIPT_DIR/neutts_env/bin/activate" ;;
  chatterbox) source "$SCRIPT_DIR/chatterbox_env/bin/activate" ;;
  qwen3)      source "$SCRIPT_DIR/qwen3_env/bin/activate" ;;   # torch + qwen-tts
                                # (pip install qwen-tts); the -VoiceDesign checkpoint downloads
                                # into the in-tree HF cache on first start (see tts_qwen3.py)
  orpheus)                      # pre-gguf configs: the vLLM engine was removed
     echo "note: engine 'orpheus' (vLLM) was retired — using orpheus_gguf" >&2
     ENGINE=orpheus_gguf; source "$SCRIPT_DIR/vinkona_env/bin/activate" ;;
  *) echo "usage: $0 {orpheus_gguf|neutts|chatterbox|qwen3}"; exit 1 ;;
esac

cd "$SCRIPT_DIR"
exec python tts_server.py --engine "$ENGINE" --config "$CONFIG"
