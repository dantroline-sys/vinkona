#!/usr/bin/env bash
# Distil ingested raw chunks into the structured KB (the meaning layer).
# Needs the big LM (distill_url) and the embed server up.  Resumable.
#   ./distill.sh                 # distil everything not yet done
#   ./distill.sh --limit 200     # cap the number of chunks (smoke test)
#   ./distill.sh --watch         # keep going as a concurrent ./ingest.sh adds chunks
#   ./distill.sh --watch --interval 60   # poll every 60s (default 30)
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
exec "$PY" -m knowledgehost distill "${ARGS[@]}" "$@"
