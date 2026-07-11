#!/usr/bin/env bash
# Clear the import so you can start from scratch (asks for a typed confirmation).
#   ./reset.sh            # clear raw chunks + distilled KB
#   ./reset.sh --kb       # clear ONLY the distilled KB (keep raw chunks → re-distill fast)
#   ./reset.sh --raw      # clear ONLY the raw chunk store (keep the KB)
#   ./reset.sh -y         # skip the confirmation prompt
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
exec "$PY" -m knowledgehost reset "${ARGS[@]}" "$@"
