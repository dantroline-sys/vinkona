#!/usr/bin/env bash
# Drain the node-merge adjudication queue with the big LM (same / distinct / is_a).
# Needs the big LM (verify/distill endpoint) up.  Resumable & lease-aware.
#   ./adjudicate.sh                 # judge the whole open queue
#   ./adjudicate.sh --limit 200     # cap pairs (chip away / smoke test)
#   ./adjudicate.sh --batch 12      # pairs per LM call (default 8)
#   ./adjudicate.sh --watch         # keep draining as distillation adds candidates
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh          # in-tree var/ caches + tmp — see env.sh
PY=python3
if [ -x ".venv/bin/python3" ]; then PY=".venv/bin/python3"; fi
ARGS=()
if [ -f config.toml ]; then ARGS+=(-c config.toml); fi
exec "$PY" -m knowledgehost adjudicate "${ARGS[@]}" "$@"
