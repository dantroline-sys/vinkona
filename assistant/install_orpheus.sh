#!/bin/bash
# Install Orpheus TTS into its OWN venv (orpheus_env).
#
# Orpheus runs a Llama-3B backbone through vLLM, which pins its own CUDA torch and
# conflicts with both neutts_env (torch 2.12) and vinkona_env (torch <2.5).
# So it's isolated here, like the other engines.  At integration each TTS runs as
# a separate local service the main server selects via a flag.
#
# Run INSIDE the distrobox (needs CUDA for vLLM).  The 3B model (~6 GB, or less
# quantized) downloads from HuggingFace on first use.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"          # in-tree caches/tmp/PATH — see env.sh
ENV_DIR="$SCRIPT_DIR/orpheus_env"

echo "== Creating isolated venv: $ENV_DIR =="
if [ ! -f "$ENV_DIR/bin/activate" ]; then
    rm -rf "$ENV_DIR"
    python3 -m venv "$ENV_DIR" || true
fi
if [ ! -f "$ENV_DIR/bin/activate" ]; then
    echo "ERROR: failed to create venv at $ENV_DIR."
    echo "Install the venv module and re-run:"
    echo "  Debian/Ubuntu: sudo apt install python3-venv   (or python3.12-venv)"
    echo "  Fedora:        sudo dnf install python3-virtualenv"
    exit 1
fi
source "$ENV_DIR/bin/activate"
pip install --upgrade pip
pip install orpheus-speech soundfile

echo ""
echo "Done — Orpheus is isolated in orpheus_env."
echo ""
echo "If you hit a vLLM error on first run, pin the known-good build:"
echo "  source orpheus_env/bin/activate && pip install vllm==0.7.3"
echo ""
echo "Smoke test (preset voice + an inline emotion tag):"
echo "  source orpheus_env/bin/activate"
echo "  CUDA_VISIBLE_DEVICES=0 python test_tts.py --engine orpheus --voice tara \\"
echo "      --text \"Well <laugh> hello there, what's on your mind?\" --out /tmp/orpheus_test.wav"
