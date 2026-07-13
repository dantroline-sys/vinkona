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
source "$SCRIPT_DIR/env.sh"          # in-tree caches/tmp/PATH — see env.sh
cd "$SCRIPT_DIR"

# glibc malloc arena bloat looks exactly like a catastrophic leak under a
# threaded server hammered with embedding batches: every thread gets its own
# arena, freed memory fragments across them and never returns to the OS, and
# RSS climbs by GB in seconds.  Two arenas is plenty for this service and
# bounds the fragmentation; export your own value to override.
: "${MALLOC_ARENA_MAX:=2}"; export MALLOC_ARENA_MAX

# HARD memory ceiling (the LAST of three defences against the llama.cpp embed
# leak — llm_server.py's graceful recycler restarts llama-server between
# requests at 50% of embed_lm.mem_max, the supervisor watchdog TERMs the
# service at 75%, and this cgroup scope lets the kernel kill it at 100%).
# Run it in its own cgroup capped at embed_lm.mem_max (default 8G) so the
# kernel kills the EMBED SERVER ALONE the instant it crosses the limit —
# the watchdog then respawns it.  Without its own cgroup, the leak builds
# until systemd-oomd kills the whole session (terminal + supervisor included),
# because oomd kills by cgroup.  VINKONA_EMBED_MEMMAX overrides config; set it
# (or config mem_max) empty to disable.  Skipped where systemd-run is absent.
MEMMAX="${VINKONA_EMBED_MEMMAX-$(python3 - 2>/dev/null <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("config", "config.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
print(mod.load_config("config/config.json").get("embed_lm", {}).get("mem_max") or "")
PY
)}"
RUN=(python3 llm_server.py --tier embed_lm --config "$SCRIPT_DIR/config/config.json")
if [ -n "$MEMMAX" ] && command -v systemd-run >/dev/null 2>&1; then
  echo "[embed] cgroup memory cap: $MEMMAX (embed_lm.mem_max; the watchdog revives on kill)"
  exec systemd-run --user --scope -p "MemoryMax=$MEMMAX" -p MemorySwapMax=0 \
       "${RUN[@]}" "$@"
fi
exec "${RUN[@]}" "$@"
