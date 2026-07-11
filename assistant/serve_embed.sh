#!/bin/bash
# Embedding LM service — semantic-recall embeddings via llama.cpp llama-server in
# --embedding mode (OpenAI /v1/embeddings).  Used by the memory system.
#
# Settings come from config/config.json (embed_lm block): GGUF, GPU, ctx_size,
# pooling.  Small and cheap — co-locate it with the fast LM on the live card.
#
#   ./serve_embed.sh                    # uses config/config.json
#   ./serve_embed.sh --dry-run          # print the llama-server command
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Optional HARD memory ceiling (complements the vinkona.sh watchdog): if VINKONA_EMBED_MEMMAX is
# set (e.g. 8G) and systemd-run is available, run the server in a cgroup scope capped at that
# resident size — the kernel kills it the instant it crosses the limit (the watchdog then
# respawns it), so a llama.cpp leak can never creep up and take the whole box with it.
RUN=(python3 llm_server.py --tier embed_lm --config "$SCRIPT_DIR/config/config.json")
if [ -n "${VINKONA_EMBED_MEMMAX:-}" ] && command -v systemd-run >/dev/null 2>&1; then
  exec systemd-run --user --scope -p "MemoryMax=${VINKONA_EMBED_MEMMAX}" -p MemorySwapMax=0 \
       "${RUN[@]}" "$@"
fi
exec "${RUN[@]}" "$@"
