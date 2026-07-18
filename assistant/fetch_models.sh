#!/bin/bash
# Download the default GGUF weights into Models/.
#
# Two sets:
#   full   — fast 3B + big 32B + embed          (GPU box / ≥32 GB machines)
#   small  — fast 3B + big 7B  + embed          (≤20 GB RAM, e.g. a 16 GB Mac mini)
# The set is picked from installed RAM unless forced with --full / --small.
# On a small machine with NO config/config.json yet, a sparse profile overlay
# is written so the runtime (which deep-merges it over DEFAULTS) actually uses
# the small big-LM and the chatterbox TTS engine — a 32B default can never
# load in 16 GB of unified memory.  An existing config.json is never touched.
#
# The filenames here match config defaults (fast_lm / big_lm / embed_lm .model).
# If you keep weights elsewhere, skip this and symlink instead:
#     ln -s /big/disk/gguf Models
# or symlink individual files into Models/.  Override the target dir with
#     MODELS_DIR=/path ./fetch_models.sh
#
# Downloads via vinkona_env's huggingface_hub python API (vk_hf_download in
# env.sh — the hub CLI's huggingface-cli->hf rename made the CLI unreliable
# to shell out to).  `git lfs` is NOT required.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"          # in-tree caches/tmp/PATH — see env.sh
DIR="${MODELS_DIR:-$SCRIPT_DIR/Models}"
mkdir -p "$DIR"

ram_gb() {
    if [ "$(uname -s)" = Darwin ]; then
        echo $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
    else
        awk '/MemTotal/ {printf "%d", $2/1048576; found=1} END {if (!found) print 0}' \
            /proc/meminfo 2>/dev/null || echo 0
    fi
}

PROFILE=""
case "${1:-}" in
    --small) PROFILE=small ;;
    --full)  PROFILE=full ;;
esac
if [ -z "$PROFILE" ]; then
    RAM="$(ram_gb)"
    if [ "$RAM" -gt 0 ] && [ "$RAM" -le 20 ]; then
        PROFILE=small
        echo "This machine has ${RAM} GB RAM — using the SMALL model set (--full to override)."
    else
        PROFILE=full
    fi
fi

# repo  filename  (one per line)
grab() {  # repo file
  echo "→ $2"
  vk_hf_download "$1" "$2" "$DIR"
}

echo "Fetching GGUFs into $DIR ($PROFILE set) ..."
# Fast LM — Qwen2.5 3B Instruct (Q4_K_M ≈ 2 GB)
grab bartowski/Qwen2.5-3B-Instruct-GGUF    Qwen2.5-3B-Instruct-Q4_K_M.gguf
if [ "$PROFILE" = small ]; then
    # Big LM (small set) — Qwen2.5 7B Instruct (Q4_K_M ≈ 4.7 GB)
    grab bartowski/Qwen2.5-7B-Instruct-GGUF    Qwen2.5-7B-Instruct-Q4_K_M.gguf
else
    # Big LM — Qwen2.5 32B Instruct (Q4_K_M ≈ 20 GB; large download)
    grab bartowski/Qwen2.5-32B-Instruct-GGUF   Qwen2.5-32B-Instruct-Q4_K_M.gguf
fi
# Embeddings — nomic-embed-text v1.5 (f16 ≈ 260 MB)
grab nomic-ai/nomic-embed-text-v1.5-GGUF   nomic-embed-text-v1.5.f16.gguf

# ── small-machine profile overlay ─────────────────────────────────────────────
# config/config.json is a sparse overlay deep-merged over config.py DEFAULTS,
# so these few keys are the whole profile: a big LM that fits, a trimmed KV
# budget, and the lightweight TTS engine (orpheus needs its own tts_lm
# llama-server on top).  Written ONLY when the user has no config yet.
if [ "$PROFILE" = small ]; then
    CFG="$SCRIPT_DIR/config/config.json"
    FRAGMENT='{
  "big_lm": {"url": "http://127.0.0.1:11438",
             "model": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
             "ctx_size": 16384},
  "tts": {"engine": "chatterbox"}
}'
    if [ ! -f "$CFG" ]; then
        mkdir -p "$SCRIPT_DIR/config"
        printf '%s\n' "$FRAGMENT" > "$CFG"
        echo "Wrote the small-machine profile to config/config.json (sparse overlay"
        echo "over defaults; edit or delete freely — the config web UI reads it too)."
    else
        echo "config/config.json exists — NOT touching it. For a small machine,"
        echo "merge this fragment yourself (or via the config web UI):"
        printf '%s\n' "$FRAGMENT"
    fi
    echo "chatterbox TTS needs its venv once:  ./install.sh tts chatterbox"
    echo "(running the knowledge host on this machine too? point vinur's LM"
    echo " endpoints at the fast/big tiers above — small models, same ports)"
    echo "(no tool host configured? the built-in ONLINE Wikipedia search is"
    echo " offered to the assistant automatically — see tools.wikipedia)"
fi

echo ""
echo "Done. Files in $DIR:"
ls -lh "$DIR"/*.gguf 2>/dev/null || true
echo ""
echo "These names match the config defaults. Start the tiers with:"
echo "  ./serve_fast_lm.sh   ./serve_big_lm.sh   ./serve_embed.sh"
echo "(verify the command first with --dry-run)"
