#!/bin/bash
# Download the default GGUF weights into Models/.
#
# The filenames here match config/config.json (fast_lm / big_lm / embed_lm .model).
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

# repo  filename  (one per line)
grab() {  # repo file
  echo "→ $2"
  vk_hf_download "$1" "$2" "$DIR"
}

echo "Fetching GGUFs into $DIR ..."
# Fast LM — Qwen2.5 3B Instruct (Q4_K_M ≈ 2 GB)
grab bartowski/Qwen2.5-3B-Instruct-GGUF    Qwen2.5-3B-Instruct-Q4_K_M.gguf
# Big LM — Qwen2.5 32B Instruct (Q4_K_M ≈ 20 GB; large download)
grab bartowski/Qwen2.5-32B-Instruct-GGUF   Qwen2.5-32B-Instruct-Q4_K_M.gguf
# Embeddings — nomic-embed-text v1.5 (f16 ≈ 260 MB)
grab nomic-ai/nomic-embed-text-v1.5-GGUF   nomic-embed-text-v1.5.f16.gguf

echo ""
echo "Done. Files in $DIR:"
ls -lh "$DIR"/*.gguf 2>/dev/null || true
echo ""
echo "These names match config/config.json. Start the tiers with:"
echo "  ./serve_fast_lm.sh   ./serve_big_lm.sh   ./serve_embed.sh"
echo "(verify the command first with --dry-run)"
