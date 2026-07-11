#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/config.json"

# For a 3090+4090 rig, prefer the 4090 (index 1) for inference throughput.
export CUDA_VISIBLE_DEVICES=1
export RUST_LOG=info

if [[ ! -f "$CONFIG" ]]; then
    echo "[ERROR] Config not found at: $CONFIG"
    echo "        Check that the model was downloaded: ls $SCRIPT_DIR/Models/"
    exit 1
fi

exec "$SCRIPT_DIR/moshi-backend" \
    --log info \
    --config "$CONFIG" \
    standalone
