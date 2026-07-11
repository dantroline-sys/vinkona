#!/bin/bash
# Install NeuTTS Air into its OWN venv (neutts_env).
#
# NeuTTS's dependencies (torch 2.12, numpy 2.2, modern huggingface-hub) are
# mutually incompatible with moshi-personaplex's pins (torch <2.5, numpy <2.2),
# so it CANNOT share vinkona_env — installing into it breaks PersonaPlex.
# We isolate NeuTTS here; at integration it runs as a separate local service the
# main server calls over HTTP (exactly like the Ollama LLM instances).
#
# Run INSIDE the distrobox.  The backbone (~748M) downloads from HF on first use.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"          # in-tree caches/tmp/PATH — see env.sh
ENV_DIR="$SCRIPT_DIR/neutts_env"

echo "== System dep: espeak-ng (G2P) =="
if ! command -v espeak-ng >/dev/null 2>&1; then
    if command -v apt >/dev/null 2>&1; then
        sudo apt install -y espeak-ng
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y espeak-ng
    else
        echo "Please install espeak-ng with your package manager."; exit 1
    fi
fi

echo "== Creating isolated venv: $ENV_DIR =="
# Recreate from scratch if there's no working activate script (a partial venv
# from a failed run, or python3-venv missing, leaves a dir with no bin/activate).
if [ ! -f "$ENV_DIR/bin/activate" ]; then
    rm -rf "$ENV_DIR"
    python3 -m venv "$ENV_DIR" || true
fi
if [ ! -f "$ENV_DIR/bin/activate" ]; then
    echo "ERROR: failed to create venv at $ENV_DIR."
    echo "The python venv module is likely missing. Install it and re-run:"
    echo "  Debian/Ubuntu: sudo apt install python3-venv   (or python3.12-venv)"
    echo "  Fedora:        sudo dnf install python3-virtualenv"
    exit 1
fi
source "$ENV_DIR/bin/activate"
pip install --upgrade pip
pip install neutts soundfile

echo ""
echo "Done — NeuTTS is isolated in neutts_env (vinkona_env untouched)."
echo ""
echo "Smoke test (needs a reference clip voices/vinkona.wav + voices/vinkona.txt):"
echo "  source neutts_env/bin/activate"
echo "  CUDA_VISIBLE_DEVICES=0 python test_tts.py --ref voices/vinkona.wav \\"
echo "      --text \"Hello, I'm Vinkona.\" --out /tmp/tts_test.wav"
