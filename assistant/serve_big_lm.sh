#!/bin/bash
# Big LM service — large model (e.g. Qwen2.5-32B) via llama.cpp llama-server.
#
# Background reasoning/briefing tier; runs between turns and at session end, so a
# little slower is fine.  Settings come from config/config.json (big_lm block):
# GGUF, GPU, ctx_size, n_gpu_layers, flash-attn, extra flags.  Recommended
# placement: dedicated on the slower card (it never blocks the live reply).
#
# Note: no more system-Ollama port clash — this binds the big_lm url itself.  If a
# stray Ollama still holds the port, stop it (systemctl stop ollama / pkill ollama).
#
#   ./serve_big_lm.sh                   # uses config/config.json
#   ./serve_big_lm.sh --dry-run         # print the llama-server command
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec python3 llm_server.py --tier big_lm --config "$SCRIPT_DIR/config/config.json" "$@"
