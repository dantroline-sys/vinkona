#!/bin/bash
# Fast LM service — small model (e.g. Qwen2.5-3B) via llama.cpp llama-server.
#
# All settings come from config/config.json (fast_lm block): which GGUF, which GPU,
# context length, n_gpu_layers, flash-attn, extra flags.  Edit them in the config
# web UI (Models tab) or the JSON.  Recommended placement: the live response path
# (fast LM + embed + TTS) on the faster card.
#
#   ./serve_fast_lm.sh                  # uses config/config.json
#   ./serve_fast_lm.sh --dry-run        # print the llama-server command
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec python3 llm_server.py --tier fast_lm --config "$SCRIPT_DIR/config/config.json" "$@"
