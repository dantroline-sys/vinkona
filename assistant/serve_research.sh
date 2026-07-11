#!/bin/bash
# Tier-3 background research worker — distils session topics into world-knowledge
# memories (Wikipedia by default).  Separate process; shares the SQLite memory
# store with the cascade via WAL.  Enable in config (research.enabled) first.
#
# v1 uses only Wikipedia over HTTP, so it runs in vinkona_env (no new deps).
#
#   ./serve_research.sh            # poll the queue forever
#   ./serve_research.sh --once     # drain what's queued and exit
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/vinkona_env/bin/activate"
cd "$SCRIPT_DIR"
exec python research_worker.py --config "$SCRIPT_DIR/config/config.json" "$@"
