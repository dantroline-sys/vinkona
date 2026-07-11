#!/bin/bash
# Second big-LM instance — for "knowledge acquisition mode" only (./vinkona.sh start knowledge).
#
# Runs the SAME large model as big_lm but on the other card (the 4090, alongside embed), so
# the knowledge-host can split its distillation across both big LMs for ~2x throughput. It
# inherits big_lm's model + llama.cpp knobs; the big_lm2 config block only overrides url
# (port 11440), gpu (the 4090), and ctx_size (small — knowledge chunks are <=2048 tokens).
#
#   ./serve_big_lm2.sh                  # uses config/config.json (big_lm2 block, inheriting big_lm)
#   ./serve_big_lm2.sh --dry-run        # print the llama-server command
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"          # in-tree caches/tmp/PATH — see env.sh
cd "$SCRIPT_DIR"
exec python3 llm_server.py --tier big_lm2 --config "$SCRIPT_DIR/config/config.json" "$@"
