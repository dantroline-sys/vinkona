#!/bin/bash
# Install a torch-based TTS engine into its OWN venv:
#
#   ./install_tts.sh neutts        -> neutts_env      (NeuTTS Air, cloned voice)
#   ./install_tts.sh chatterbox    -> chatterbox_env  (Chatterbox, ~0.5B — the
#                                     low-footprint choice for small machines)
#
# Both stacks carry torch and tight pins; vinkona_env is deliberately
# torch-free, so each engine gets its own uv project (deps/<engine>/) and its
# own venv, and runs as a separate local service the main server calls over
# HTTP (exactly like the local LLM instances).
#
# On container setups run INSIDE the distrobox; on macOS or a plain host, run
# directly.  Model weights download from the HF hub on first use (in-tree).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"          # in-tree caches/tmp/PATH — see env.sh

ENGINE="${1:-neutts}"
case "$ENGINE" in
    neutts)
        ENV_DIR="$SCRIPT_DIR/neutts_env"
        echo "== System dep: espeak-ng (G2P) =="
        vk_require_tools espeak-ng || exit 1
        ;;
    chatterbox)
        ENV_DIR="$SCRIPT_DIR/chatterbox_env"
        ;;
    *) echo "usage: $0 {neutts|chatterbox}"; exit 1 ;;
esac

echo "== Isolated venv: $ENV_DIR (uv sync from deps/$ENGINE — its own project, none of the core deps) =="
# Each engine has its own uv project (deps/<engine>/pyproject.toml + lock):
# torch stacks cap numpy and lag new CPython releases, so they must not share
# a resolution with vinkona_env — or with each other. If the system Python is
# too new for one, uv downloads a matching CPython into var/uv/python — no
# PYTHON= hunting (though PYTHON=3.12 / a name / a path still overrides).
UVARGS=(sync --inexact --project "$SCRIPT_DIR/deps/$ENGINE")
[ -n "${PYTHON:-}" ] && UVARGS+=(--python "$PYTHON")
UV_PROJECT_ENVIRONMENT="$ENV_DIR" vk_uv "${UVARGS[@]}" \
    || { echo "ERROR: uv sync failed — see above."; exit 1; }

echo "== Verifying the install (a failed sync must never look like a green tick) =="
case "$ENGINE" in
    neutts)     "$ENV_DIR/bin/python" -c "import numpy, soundfile; print('neutts_env sanity: numpy + soundfile present')" ;;
    chatterbox) "$ENV_DIR/bin/python" -c "import chatterbox, numpy, soundfile; print('chatterbox_env sanity: chatterbox + numpy + soundfile present')" ;;
esac

echo ""
echo "Done — $ENGINE is isolated in $(basename "$ENV_DIR") (vinkona_env untouched)."
echo ""
case "$ENGINE" in
    neutts)
        echo "Smoke test (needs a reference clip voices/vinkona.wav + voices/vinkona.txt):"
        echo "  source neutts_env/bin/activate"
        echo "  python test_tts.py --engine neutts --ref voices/vinkona.wav \\"
        echo "      --text \"Hello, I'm Vinkona.\" --out /tmp/tts_test.wav"
        ;;
    chatterbox)
        echo "Smoke test (no reference needed — the built-in voice; add --ref clip.wav to clone):"
        echo "  source chatterbox_env/bin/activate"
        echo "  python test_tts.py --engine chatterbox --text \"Hello, I'm Vinkona.\" --out /tmp/tts_test.wav"
        echo ""
        echo "To make it the active engine: set tts.engine = \"chatterbox\" (config UI), restart."
        ;;
esac
