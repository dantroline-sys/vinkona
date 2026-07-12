#!/bin/bash
# TTS LM service — the Orpheus 3B GGUF backbone via llama.cpp llama-server.
#
# Only needed when config tts.engine is "orpheus_gguf": tts_server.py (the
# orpheus_gguf engine) streams audio tokens from this server and vocodes them
# on the CPU.  All settings come from config/config.json (tts_lm block): which
# GGUF, which GPU, context length.  vinkona.sh starts this automatically when
# the engine is orpheus_gguf.
#
#   ./serve_tts_lm.sh                  # uses config/config.json
#   ./serve_tts_lm.sh --dry-run        # print the llama-server command
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"          # in-tree caches/tmp/PATH — see env.sh
cd "$SCRIPT_DIR"
exec python3 llm_server.py --tier tts_lm --config "$SCRIPT_DIR/config/config.json" "$@"
