#!/bin/bash
# Download the default GGUF weights into Models/ via huggingface-cli.
#
# The filenames here match config/config.json (fast_lm / big_lm / embed_lm .model).
# If you keep weights elsewhere, skip this and symlink instead:
#     ln -s /big/disk/gguf Models
# or symlink individual files into Models/.  Override the target dir with
#     MODELS_DIR=/path ./fetch_models.sh
#
# Uses vinkona_env's huggingface-cli (installed by ./install.sh core), falling
# back to a system-wide one.  `git lfs` is NOT required.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"          # in-tree caches/tmp/PATH — see env.sh
DIR="${MODELS_DIR:-$SCRIPT_DIR/Models}"
mkdir -p "$DIR"

HF_CLI="$SCRIPT_DIR/vinkona_env/bin/huggingface-cli"
if [ ! -x "$HF_CLI" ]; then
  HF_CLI="$(command -v huggingface-cli || true)"
fi
if [ -z "$HF_CLI" ]; then
  echo "huggingface-cli not found — run './install.sh core' first (it installs" >&2
  echo "huggingface_hub into vinkona_env), or: pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

# repo  filename  (one per line)
grab() {  # repo file
  echo "→ $2"
  "$HF_CLI" download "$1" "$2" --local-dir "$DIR" --local-dir-use-symlinks False >/dev/null
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
