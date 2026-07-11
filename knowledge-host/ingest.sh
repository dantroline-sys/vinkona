#!/usr/bin/env bash
# Run the offline ingestion pipeline (heavy/batch — run on demand or monthly).
#   ./ingest.sh                      # crawl the configured document roots
#   ./ingest.sh --wikipedia          # also ingest the configured Wikipedia ZIM
#   ./ingest.sh --wikipedia --limit 500   # cap ZIM articles (smoke test)
#   ./ingest.sh --force              # re-process every file (ignore manifest)
#   ./ingest.sh --distill            # distil the newly-ingested chunks right after
# (distillation is also a standalone step: ./distill.sh — see that script)
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh          # in-tree var/ caches + tmp — see env.sh

# Prefer the virtualenv built by install.sh; fall back to system python3.
PY=python3
if [ -x ".venv/bin/python3" ]; then PY=".venv/bin/python3"; fi

ARGS=()
if [ -f config.toml ]; then
  ARGS+=(-c config.toml)
fi
exec "$PY" -m knowledgehost ingest "${ARGS[@]}" "$@"
