#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/vinkona_env/bin/activate"

# Kill any stale moshi server left over from a previous crash.
# TERM first so CUDA streams can flush; KILL after 3 s if still alive.
if pgrep -f "moshi.server" > /dev/null 2>&1; then
    echo "Killing stale moshi.server process(es)..."
    pkill -TERM -f "moshi.server"
    sleep 3
    pkill -KILL -f "moshi.server" 2>/dev/null || true
    # Give the GPU driver a moment to release the CUDA context.
    sleep 1
fi

MODEL_DIR="$SCRIPT_DIR/Models/personaplex-7b-v1-raw"

# Use the pre-generated certs (cert.pem + key.pem must exist here)
SSL_DIR="$SCRIPT_DIR/certs"

# For a 3090+4090 rig, prefer the 4090 (index 1) for inference throughput.
export CUDA_VISIBLE_DEVICES=1

# Disable torch.compile — PyTorch 2.9 inductor has internal import inconsistencies
# that crash on first compilation attempt. moshi/utils/compile.py checks this flag
# at decoration time and returns bare functions, so no torch.compile call is made.
export NO_TORCH_COMPILE=1

# Ensure the real libcuda.so (from the host NVIDIA driver) is found at runtime.
# The distrobox ldconfig cache may not include this path, causing Triton to fail
# with "undefined symbol: cuModuleGetFunction" when loading compiled kernels.
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

python -m moshi.server \
  --host 0.0.0.0 \
  --port 8998 \
  --ssl "$SSL_DIR" \
  --config-path  "$SCRIPT_DIR/personaplex_config.json" \
  --moshi-weight "$MODEL_DIR/model.safetensors" \
  --mimi-weight  "$MODEL_DIR/tokenizer-e351c8d8-checkpoint125.safetensors" \
  --tokenizer    "$MODEL_DIR/tokenizer_spm_32k_3.model" \
  --hf-repo      nvidia/personaplex-7b-v1 \
  --mic-gain     4.0 \
  --denoise \
  --asr \
  --fast-lm-url   http://127.0.0.1:11435 \
  --fast-model    qwen2.5:3b \
  --personas      "$SCRIPT_DIR/personas.json"
  # ── Big LM (optional, GPU 0 / 4090) — add once the fast tier is proven ────
  # --big-lm-url   http://127.0.0.1:11434 \
  # --big-model    qwen2.5:32b \
  # ── Other options ─────────────────────────────────────────────────────────
  # --loopback   # audio pipeline test, bypasses LM
