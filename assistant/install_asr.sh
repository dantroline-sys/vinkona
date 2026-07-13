#!/bin/bash
# Install faster-whisper (CPU ASR) into vinkona_env.
# Run INSIDE the distrobox.  CTranslate2 ships CPU wheels — no compiler needed.
# The model itself (e.g. base.en ~140 MB) downloads from HuggingFace on first use.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"          # in-tree caches/tmp/PATH — see env.sh
source "$SCRIPT_DIR/vinkona_env/bin/activate"

pip install faster-whisper soxr

echo ""
echo "Done.  Verify with:"
echo "  python -c \"from faster_whisper import WhisperModel; print('faster-whisper OK')\""
echo ""
echo "Enable in serve.sh by adding:  --asr   (optionally --asr-model small.en)"
