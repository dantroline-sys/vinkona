#!/bin/bash
# Verify faster-whisper (CPU ASR) in vinkona_env — it's part of the core set,
# so this syncs the environment (uv) and proves the import works.
# On container setups run INSIDE the distrobox; on macOS or a plain host, run
# it directly.  CTranslate2 ships CPU wheels — no compiler needed.  The model
# itself (e.g. base.en ~140 MB) downloads from HuggingFace on first use.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"          # in-tree caches/tmp/PATH — see env.sh

UV_PROJECT_ENVIRONMENT="$SCRIPT_DIR/vinkona_env" vk_uv sync --inexact --project "$SCRIPT_DIR"
"$SCRIPT_DIR/vinkona_env/bin/python" -c \
    "from faster_whisper import WhisperModel; import soxr; print('faster-whisper + soxr OK')"
