#!/bin/bash
# Start the config web UI (localhost only by default — see config_server.host).
# Edits config/config.json and config/personas.json; the cascade picks up persona
# and tunable changes on the next call (ports/models need a restart).
#
#   ./serve_config.sh           then open http://127.0.0.1:8090
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/vinkona_env/bin/activate"
cd "$SCRIPT_DIR"
exec python config_server.py --config "$SCRIPT_DIR/config/config.json"
