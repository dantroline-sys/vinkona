#!/bin/bash
# Start the TTS service in the correct isolated venv for the chosen engine.
#
#   ./serve_tts.sh orpheus      # uses orpheus_env
#   ./serve_tts.sh neutts       # uses neutts_env
#
# All settings (port, voice, model, gpu mem, refs) come from config/config.json;
# the engine arg only selects which venv to activate.  Override the GPU with
# CUDA_VISIBLE_DEVICES=N ./serve_tts.sh ...
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"          # in-tree caches/tmp/PATH — see env.sh
ENGINE="${1:-orpheus}"
CONFIG="$SCRIPT_DIR/config/config.json"

# Stable GPU ordering so the LM servers (llama.cpp) and vLLM agree on indices.
# The live response path is fast LM + embed + TTS together on the 4090; the big LM
# runs alone on the 3090.  On this box the 4090 is CUDA index 1 (verify with
# CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi).  Override with
# CUDA_VISIBLE_DEVICES=N ./serve_tts.sh ...
export CUDA_DEVICE_ORDER=PCI_BUS_ID
: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES

case "$ENGINE" in
  orpheus) source "$SCRIPT_DIR/orpheus_env/bin/activate" ;;
  neutts)  source "$SCRIPT_DIR/neutts_env/bin/activate" ;;
  *) echo "usage: $0 {orpheus|neutts}"; exit 1 ;;
esac

cd "$SCRIPT_DIR"
exec python tts_server.py --engine "$ENGINE" --config "$CONFIG"
