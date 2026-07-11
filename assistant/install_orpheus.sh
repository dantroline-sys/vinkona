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

# vLLM's dependency chain (numba) currently supports Python >=3.10,<3.14 — a
# newer system python (e.g. Fedora 44's 3.14) cannot build this venv. Pick a
# supported interpreter (override with PYTHON=...), offering to install one
# via the distro package manager if none is present.
PY_MIN=10; PY_MAX=13
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    PY="$(vk_pick_python "$PY_MIN" "$PY_MAX" || true)"
fi
if [ -z "$PY" ]; then
    echo "Orpheus/vLLM needs Python 3.$PY_MIN–3.$PY_MAX; your python3 is $(python3 -V 2>/dev/null | awk '{print $2}' || echo 'not found')."
    vk_require_tools "python3.13:python3.13" || exit 1
    PY=python3.13
fi
echo "== Interpreter: $PY =="

# A venv left behind by an earlier run on an unsupported interpreter (e.g. a
# failed install on 3.14) must be rebuilt, not reused.
if [ -x "$ENV_DIR/bin/python" ]; then
    have="$("$ENV_DIR/bin/python" -c 'import sys; print(sys.version_info[1])' 2>/dev/null || echo 0)"
    if [ "$have" -lt "$PY_MIN" ] || [ "$have" -gt "$PY_MAX" ]; then
        echo "existing orpheus_env is on python3.$have (unsupported) — recreating with $PY"
        rm -rf "$ENV_DIR"
    fi
fi

echo "== Creating isolated venv: $ENV_DIR =="
if [ ! -f "$ENV_DIR/bin/activate" ]; then
    rm -rf "$ENV_DIR"
    "$PY" -m venv "$ENV_DIR" || true
fi
if [ ! -f "$ENV_DIR/bin/activate" ]; then
    echo "ERROR: failed to create venv at $ENV_DIR with $PY."
    echo "Install the venv module for it and re-run:"
    echo "  Debian/Ubuntu: sudo apt install ${PY}-venv"
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
