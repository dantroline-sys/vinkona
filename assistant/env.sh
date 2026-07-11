# Vinkona filesystem confinement — source this from every script (serve_*, install*).
#
# THE GUARANTEE: everything Vinkona writes stays inside this folder tree.
# This file is how that's enforced for the third-party stacks we run: model
# downloads (huggingface_hub / faster-whisper / vLLM), compile caches
# (torch inductor, triton), and temp files are all pinned under ./var.
#
# Vinkona's own writes already live in-tree: config/ (live config, personas,
# memory.db, ws token), logs/, Models/. With this file, so does everything else:
#
#   var/cache/     third-party caches (HF hub, torch, vLLM, triton, pip, …)
#   var/tmp/       temp files (TMPDIR)
#   var/build/     source builds (rnnoise, llama.cpp)
#   var/rnnoise/   the in-tree librnnoise install prefix
#   bin/           in-tree binaries (llama-server if built by ./install.sh llama)
#
# Reads are unrestricted — point Models/ or config at anything you like.
# Known exceptions (documented, not writes by Vinkona itself): system packages
# you install with dnf/apt (espeak-ng, toolchain), and tmux's own socket.
#
# Everything here is process-scoped: sourcing this affects Vinkona's services
# only, never your shell profile or other programs.

VINKONA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VINKONA_ROOT
export VINKONA_VAR="$VINKONA_ROOT/var"

# Model hubs + ML caches
export HF_HOME="$VINKONA_VAR/cache/huggingface"
export TORCH_HOME="$VINKONA_VAR/cache/torch"
export VLLM_CACHE_ROOT="$VINKONA_VAR/cache/vllm"
export TRITON_CACHE_DIR="$VINKONA_VAR/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$VINKONA_VAR/cache/torchinductor"
export NUMBA_CACHE_DIR="$VINKONA_VAR/cache/numba"

# Anything that honours XDG (pip's cache, misc libraries)
export XDG_CACHE_HOME="$VINKONA_VAR/cache"

# Temp files — pip build isolation, vLLM compile scratch, OCR, …
export TMPDIR="$VINKONA_VAR/tmp"

# In-tree binaries (llama-server) take precedence, system PATH still works
export PATH="$VINKONA_ROOT/bin:$PATH"

mkdir -p "$VINKONA_VAR/cache" "$VINKONA_VAR/tmp"
