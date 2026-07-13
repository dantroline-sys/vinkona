#!/bin/bash
# Install NeuTTS Air into its OWN venv (neutts_env).
#
# NeuTTS needs torch; vinkona_env is deliberately torch-free (pure wheels, fast
# installs), so NeuTTS gets its own venv and runs as a separate local service
# the main server calls over HTTP (exactly like the local LLM instances).
#
# Run INSIDE the distrobox.  The backbone (~748M) downloads from HF on first use.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"          # in-tree caches/tmp/PATH — see env.sh
ENV_DIR="$SCRIPT_DIR/neutts_env"

echo "== System dep: espeak-ng (G2P) =="
vk_require_tools espeak-ng || exit 1

echo "== Isolated venv: $ENV_DIR (uv sync from deps/neutts — its own project, none of the core deps) =="
# NeuTTS has its own uv project (deps/neutts/pyproject.toml + lock): its
# torch/numba stack caps numpy and lags new CPython releases, so it must not
# share a resolution with vinkona_env. If the system Python is too new for it,
# uv downloads a matching CPython into var/uv/python — no PYTHON= hunting
# (though PYTHON=3.13 / a name / a path still overrides).
UVARGS=(sync --inexact --project "$SCRIPT_DIR/deps/neutts")
[ -n "${PYTHON:-}" ] && UVARGS+=(--python "$PYTHON")
UV_PROJECT_ENVIRONMENT="$ENV_DIR" vk_uv "${UVARGS[@]}" \
    || { echo "ERROR: uv sync failed — see above."; exit 1; }

echo "== Verifying the install (a failed pip must never look like a green tick) =="
"$ENV_DIR/bin/python" -c "import numpy, soundfile; print('neutts_env sanity: numpy + soundfile present')"

echo ""
echo "Done — NeuTTS is isolated in neutts_env (vinkona_env untouched)."
echo ""
echo "Smoke test (needs a reference clip voices/vinkona.wav + voices/vinkona.txt):"
echo "  source neutts_env/bin/activate"
echo "  CUDA_VISIBLE_DEVICES=0 python test_tts.py --ref voices/vinkona.wav \\"
echo "      --text \"Hello, I'm Vinkona.\" --out /tmp/tts_test.wav"
