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

# ── vk_pick_python: newest CPython 3.x on PATH within [min,max] minors ──────
# Usage:  PY="$(vk_pick_python 10 13 || true)"   # empty if none qualifies
# Prefers plain python3 when it qualifies (fewest surprises), then scans
# newest-first. For stacks whose deps cap the interpreter (vLLM/numba).
vk_pick_python() {
    local min="$1" max="$2" x m
    if command -v python3 >/dev/null 2>&1; then
        m="$(python3 -c 'import sys; print(sys.version_info[1])' 2>/dev/null)" || m=""
        if [ -n "$m" ] && [ "$m" -ge "$min" ] && [ "$m" -le "$max" ]; then
            echo python3; return 0
        fi
    fi
    for ((x=max; x>=min; x--)); do
        command -v "python3.$x" >/dev/null 2>&1 && { echo "python3.$x"; return 0; }
    done
    return 1
}

# ── vk_require_tools: check for system tools, offer to install the missing ──
# Usage:   vk_require_tools gcc make "libtoolize:libtool" "g++:gcc-c++|g++" || exit 1
# Spec:    tool[:package] — package may be "dnfname|aptname" where they differ.
# Interactive shells get a [Y/n] offer to run sudo dnf/apt themselves;
# non-interactive shells get the exact command printed and a non-zero return.
vk_require_tools() {
    local mgr="" pick=1 spec tool pkg missing=() pkgs=()
    if command -v dnf >/dev/null 2>&1;      then mgr="dnf install -y"
    elif command -v apt-get >/dev/null 2>&1; then mgr="apt-get install -y"; pick=2
    elif command -v pacman >/dev/null 2>&1;  then mgr="pacman -S --needed --noconfirm"
    elif command -v zypper >/dev/null 2>&1;  then mgr="zypper install -y"
    fi
    for spec in "$@"; do
        tool="${spec%%:*}"
        command -v "$tool" >/dev/null 2>&1 && continue
        pkg="${spec#*:}"; [ "$pkg" = "$spec" ] && pkg="$tool"
        if [ "$pick" -eq 2 ]; then pkg="${pkg##*|}"; else pkg="${pkg%%|*}"; fi
        missing+=("$tool"); pkgs+=("$pkg")
    done
    [ "${#missing[@]}" -eq 0 ] && return 0
    echo "Missing system tools: ${missing[*]}"
    if [ -z "$mgr" ]; then
        echo "No known package manager found — install them yourself, then re-run."
        return 1
    fi
    if [ -t 0 ] || [ "${VINKONA_ASSUME_TTY:-}" = 1 ]; then
        printf "Install now with 'sudo %s %s'? [Y/n]: " "$mgr" "${pkgs[*]}"
        local answer; read -r answer
        case "$answer" in
            n*|N*) echo "Skipped — install them and re-run."; return 1 ;;
        esac
        # shellcheck disable=SC2086
        sudo $mgr "${pkgs[@]}" || { echo "Package install failed — install manually, then re-run."; return 1; }
        hash -r
        local t
        for t in "${missing[@]}"; do
            command -v "$t" >/dev/null 2>&1 || { echo "Still missing after install: $t"; return 1; }
        done
        echo "System tools installed."
        return 0
    fi
    echo "Non-interactive shell — run:  sudo $mgr ${pkgs[*]}   then re-run this script."
    return 1
}

# ── vk_hf_download: fetch one file from the HuggingFace hub ──────────────────
# Usage: vk_hf_download <repo> <file> <dest-dir>
# Deliberately uses the PYTHON API (hf_hub_download), never the CLI: the hub's
# command-line tool was renamed huggingface-cli -> hf and the old name now
# hard-fails, which broke installs mid-flight. The python function has kept the
# same signature throughout, so this works on any huggingface_hub version.
# Subpaths in <file> are preserved under <dest-dir>; progress shows on stderr.
vk_hf_download() {
    local py="$VINKONA_ROOT/vinkona_env/bin/python"
    [ -x "$py" ] || py=python3
    "$py" - "$1" "$2" "$3" <<'PY'
import sys
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.exit("huggingface_hub is not installed — run './install.sh core' first "
             "(it goes into vinkona_env), or: pip install huggingface_hub")
repo, filename, dest = sys.argv[1:4]
path = hf_hub_download(repo_id=repo, filename=filename, local_dir=dest)
print(f"  -> {path}")
PY
}
