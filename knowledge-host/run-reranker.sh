#!/usr/bin/env bash
# Launch the local cross-encoder reranker the Knowledge Host calls (rerank=cross-encoder).
#
#   ./run-reranker.sh                 # download the model if needed, then serve on :11436
#   PORT=11439 ./run-reranker.sh      # different port (11437 is the embed server)
#   LLAMA_SERVER=/path/to/llama-server ./run-reranker.sh
#
# CPU-only by design (-ngl 0): a 568M reranker scoring a ~64-passage shortlist is
# comfortable on CPU and leaves the GPU to the cascade. Exposes Jina-style
# /rerank, which knowledgehost/rerank.py speaks; if this isn't running, search
# falls back to the in-process heuristic reranker.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh          # in-tree var/ caches + tmp — see env.sh

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-11439}"                 # next to the embed server on 11437; 11436=tts, 11438=big_lm
MODEL_DIR="${MODEL_DIR:-models}"
MODEL_FILE="${MODEL_FILE:-bge-reranker-v2-m3-Q8_0.gguf}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/resolve/main/${MODEL_FILE}}"
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/${MODEL_FILE}}"

# Locate the llama.cpp server binary (override with LLAMA_SERVER=...).
BIN="${LLAMA_SERVER:-}"
if [ -z "$BIN" ]; then BIN="$(command -v llama-server || true)"; fi
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
  echo "error: llama-server not found. Set LLAMA_SERVER=/path/to/llama-server." >&2
  exit 1
fi

# Fetch the GGUF once (resumable); rerankers are quant-sensitive, so Q8_0.
if [ ! -f "$MODEL_PATH" ]; then
  echo "downloading $MODEL_FILE (~606 MB) -> $MODEL_PATH"
  mkdir -p "$MODEL_DIR"
  curl -fL -C - --retry 3 -o "$MODEL_PATH" "$MODEL_URL"
fi

echo "reranker: $BIN  model=$MODEL_PATH  http://${HOST}:${PORT}/rerank  (cpu)"
# --reranking exposes /rerank (+ /v1/rerank). -ub/-b sized so a query+passage
# sequence never overflows the physical batch (the embed-server gotcha).
exec "$BIN" --reranking --pooling rank \
  -m "$MODEL_PATH" \
  -ngl 0 -c 4096 -b 2048 -ub 2048 \
  --host "$HOST" --port "$PORT" "$@"
