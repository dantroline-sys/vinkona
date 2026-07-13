# Vinkona filesystem confinement — source this from every script (serve_*, install*).
#
# THE GUARANTEE: everything Vinkona writes stays inside this folder tree.
# This file is how that's enforced for the third-party stacks we run: model
# downloads (huggingface_hub / faster-whisper / torch), compile caches
# (torch inductor, triton), and temp files are all pinned under ./var.
#
# Vinkona's own writes already live in-tree: config/ (live config, personas,
# memory.db, ws token), logs/, Models/. With this file, so does everything else:
#
#   var/cache/     third-party caches (HF hub, torch, triton, pip, …)
#   var/tmp/       temp files (TMPDIR)
#   var/build/     source builds (rnnoise, llama.cpp)
#   var/rnnoise/   the in-tree librnnoise install prefix
#   bin/           in-tree binaries (llama-server if built by ./install.sh llama)
#
# Reads are unrestricted — point Models/ or config at anything you like.
# Known exceptions (documented, not writes by Vinkona itself): system packages
# you install with dnf/apt/brew (espeak-ng, toolchain).
#
# Everything here is process-scoped: sourcing this affects Vinkona's services
# only, never your shell profile or other programs.

VINKONA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VINKONA_ROOT
export VINKONA_VAR="$VINKONA_ROOT/var"

# Model hubs + ML caches
export HF_HOME="$VINKONA_VAR/cache/huggingface"
export TORCH_HOME="$VINKONA_VAR/cache/torch"
export TRITON_CACHE_DIR="$VINKONA_VAR/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$VINKONA_VAR/cache/torchinductor"
export NUMBA_CACHE_DIR="$VINKONA_VAR/cache/numba"

# Anything that honours XDG (pip's cache, misc libraries)
export XDG_CACHE_HOME="$VINKONA_VAR/cache"

# uv (the python env manager — see vk_uv below): wheel cache + any CPython
# interpreters it downloads both stay in-tree. Without the second line,
# downloaded interpreters would land in ~/.local/share/uv.
export UV_CACHE_DIR="$VINKONA_VAR/cache/uv"
export UV_PYTHON_INSTALL_DIR="$VINKONA_VAR/uv/python"

# Temp files — pip build isolation, compile scratch, OCR, …
export TMPDIR="$VINKONA_VAR/tmp"

# In-tree binaries (llama-server) take precedence, system PATH still works
export PATH="$VINKONA_ROOT/bin:$PATH"

mkdir -p "$VINKONA_VAR/cache" "$VINKONA_VAR/tmp"

# ── vk_uv: run uv, bootstrapping it in-tree on first use ────────────────────
# uv (https://docs.astral.sh/uv/) builds the venvs from pyproject.toml + uv.lock
# — same pinned set on every platform, and it downloads a matching CPython
# itself (into var/uv/python) if the system one doesn't satisfy requires-python,
# so there is no more "which python3 does this machine have" problem.
# A system-wide uv is used when present; otherwise one standalone binary is
# fetched into ./bin (UV_UNMANAGED_INSTALL = no PATH/rc edits, no self-update
# state — in-tree like everything else). The venvs it makes are plain venvs.
vk_uv() {
    local uv
    uv="$(command -v uv 2>/dev/null || true)"
    [ -n "$uv" ] || uv="$VINKONA_ROOT/bin/uv"
    if [ ! -x "$uv" ]; then
        vk_require_tools curl || return 1
        echo "==> fetching uv (one-time, into bin/)" >&2
        mkdir -p "$VINKONA_ROOT/bin"
        curl -LsSf https://astral.sh/uv/install.sh \
                | env UV_UNMANAGED_INSTALL="$VINKONA_ROOT/bin" sh >&2 \
            || { echo "could not bootstrap uv — install it yourself (https://docs.astral.sh/uv/) and re-run" >&2; return 1; }
        uv="$VINKONA_ROOT/bin/uv"
    fi
    "$uv" "$@"
}

# ── vk_require_tools: check for system tools, offer to install the missing ──
# Usage:   vk_require_tools gcc make "libtoolize:libtool" "g++:gcc-c++|g++" || exit 1
# Spec:    tool[:package] — package may be "dnfname|aptname|brewname" where they
#          differ; a spec with fewer alternatives falls back to the first name.
# Interactive shells get a [Y/n] offer to run the package manager themselves
# (sudo dnf/apt/…; Homebrew on macOS runs WITHOUT sudo — it refuses root);
# non-interactive shells get the exact command printed and a non-zero return.
vk_require_tools() {
    local mgr="" pick=1 sudo_cmd="sudo" spec tool pkg a1 a2 a3 missing=() pkgs=()
    if command -v dnf >/dev/null 2>&1;      then mgr="dnf install -y"
    elif command -v apt-get >/dev/null 2>&1; then mgr="apt-get install -y"; pick=2
    elif command -v brew >/dev/null 2>&1;    then mgr="brew install"; pick=3; sudo_cmd=""
    elif command -v pacman >/dev/null 2>&1;  then mgr="pacman -S --needed --noconfirm"
    elif command -v zypper >/dev/null 2>&1;  then mgr="zypper install -y"
    fi
    for spec in "$@"; do
        tool="${spec%%:*}"
        command -v "$tool" >/dev/null 2>&1 && continue
        pkg="${spec#*:}"; [ "$pkg" = "$spec" ] && pkg="$tool"
        IFS='|' read -r a1 a2 a3 <<<"$pkg"
        case "$pick" in
            2) pkg="${a2:-$a1}" ;;
            3) pkg="${a3:-$a1}" ;;
            *) pkg="$a1" ;;
        esac
        missing+=("$tool"); pkgs+=("$pkg")
    done
    [ "${#missing[@]}" -eq 0 ] && return 0
    echo "Missing system tools: ${missing[*]}"
    if [ -z "$mgr" ]; then
        echo "No known package manager found — install them yourself, then re-run."
        return 1
    fi
    if [ -t 0 ] || [ "${VINKONA_ASSUME_TTY:-}" = 1 ]; then
        printf "Install now with '%s%s %s'? [Y/n]: " "${sudo_cmd:+$sudo_cmd }" "$mgr" "${pkgs[*]}"
        local answer; read -r answer
        case "$answer" in
            n*|N*) echo "Skipped — install them and re-run."; return 1 ;;
        esac
        # shellcheck disable=SC2086
        $sudo_cmd $mgr "${pkgs[@]}" || { echo "Package install failed — install manually, then re-run."; return 1; }
        hash -r
        local t
        for t in "${missing[@]}"; do
            command -v "$t" >/dev/null 2>&1 || { echo "Still missing after install: $t"; return 1; }
        done
        echo "System tools installed."
        return 0
    fi
    echo "Non-interactive shell — run:  ${sudo_cmd:+$sudo_cmd }$mgr ${pkgs[*]}   then re-run this script."
    return 1
}

# ── vk_ncpu: portable CPU count for make -j (nproc is Linux-only) ────────────
vk_ncpu() { nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4; }

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
             "(it goes into vinkona_env)")
repo, filename, dest = sys.argv[1:4]
path = hf_hub_download(repo_id=repo, filename=filename, local_dir=dest)
print(f"  -> {path}")
PY
}
