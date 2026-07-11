#!/bin/bash
# Memory system deps: just the embedding model GGUF for semantic recall.
#
# The store itself is stdlib SQLite + numpy (already in vinkona_env) and a
# pure-Python Aho-Corasick index — nothing to install there.  Semantic recall
# uses the embed LM tier (config embed_lm), a llama.cpp llama-server in
# --embedding mode serving a small embedding GGUF.
#
# This just checks the embedding GGUF is present in Models/ (fetch_models.sh
# downloads it along with the chat models).  Then run ./serve_embed.sh.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"          # in-tree caches/tmp/PATH — see env.sh
DIR="${MODELS_DIR:-$SCRIPT_DIR/Models}"
EMBED="nomic-embed-text-v1.5.f16.gguf"      # must match config embed_lm.model

if [ -e "$DIR/$EMBED" ]; then
  echo "Embedding model present: $DIR/$EMBED"
else
  echo "Embedding model missing. Fetch it with:"
  echo "  pip install -U 'huggingface_hub[cli]'"
  echo "  huggingface-cli download nomic-ai/nomic-embed-text-v1.5-GGUF $EMBED --local-dir '$DIR' --local-dir-use-symlinks False"
  echo "(or run ./fetch_models.sh to get all three models)"
  exit 1
fi

echo ""
echo "Start the embed server, then verify it answers:"
echo "  ./serve_embed.sh"
echo "  curl -s http://127.0.0.1:11437/v1/embeddings -H 'Content-Type: application/json' \\"
echo "    -d '{\"input\":\"hello\"}' | head -c 120"
echo ""
echo "Memory is enabled in config/config.json (memory.enabled); store at config/memory.db."
