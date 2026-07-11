#!/usr/bin/env bash
# Start the Knowledge Host query service. Edit config.toml (copy from
# config.example.toml) or override with env vars / flags. Examples:
#   ./run.sh                                  # uses ./config.toml if present
#   ./run.sh --port 8771 --backend sqlite
#   KNOWLEDGEHOST_AUTH_TOKEN=secret ./run.sh
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
exec "$PY" -m knowledgehost "${ARGS[@]}" "$@"
