#!/bin/bash
# Start the cascade voice server (the realtime voice loop).
#
# Prereqs, each in its own process (settings from config/config.json):
#   ./serve_fast_lm.sh           # fast LM   — llama.cpp (GPU 0)
#   ./serve_big_lm.sh            # big LM    — llama.cpp (GPU 1, optional)
#   ./serve_embed.sh             # embed LM  — llama.cpp (GPU 0, for memory)
#   ./serve_tts.sh orpheus_gguf  # TTS service (see serve_tts_lm.sh)
#   ./serve_tunnel.sh            # SSH tunnel to the Mac tool host (if tools.tunnel on)
# Then this, in vinkona_env (rnnoise/soxr/faster-whisper/aiohttp):
#   ./serve_cascade.sh
#
# All settings come from config/config.json.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"          # in-tree caches/tmp/PATH — see env.sh
source "$SCRIPT_DIR/vinkona_env/bin/activate"

# Real libcuda for faster-whisper / ctranslate2 CUDA, same as serve.sh.
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

cd "$SCRIPT_DIR"
exec python cascade_server.py --config "$SCRIPT_DIR/config/config.json"
