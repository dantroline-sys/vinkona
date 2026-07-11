#!/bin/bash
# Vinkona assistant — installer / uninstaller.
#
# Everything this script (and the stack it installs) writes stays INSIDE this
# folder tree: venvs, model weights, caches, builds, logs, config. No sudo, no
# /usr/local, no ~/.cache. See env.sh for how that's enforced, and the README's
# "Filesystem guarantee" section for the full contract.
#
# Usage:
#   ./install.sh                   # core: vinkona_env + cascade/ASR/memory deps + rnnoise
#   ./install.sh tts orpheus       # Orpheus TTS in its own venv (vLLM; needs CUDA)
#   ./install.sh tts neutts        # NeuTTS in its own venv
#   ./install.sh models            # download the default GGUFs into Models/
#   ./install.sh llama             # build llama.cpp's llama-server into ./bin
#   ./install.sh all               # core + tts orpheus + models (+ llama if absent)
#   ./install.sh status            # what's installed and how big it is
#   ./install.sh uninstall         # remove everything generated (venvs, var/, bin/)
#                 --with-models    #   also delete downloaded weights in Models/
#                 --purge          #   ALSO delete user data (config/, logs/) — asks first
#
# Components are also standalone scripts (install_asr.sh, install_orpheus.sh,
# install_rnnoise.sh, fetch_models.sh, …) — this orchestrates them in the right
# order. Re-running any step is safe and incremental.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"          # in-tree caches/tmp/PATH — see env.sh
cd "$SCRIPT_DIR"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; RESET='\033[0m'
say()  { echo -e "${CYAN}==>${RESET} $*"; }
ok()   { echo -e "${GREEN}ok:${RESET}  $*"; }
warn() { echo -e "${YELLOW}warning:${RESET} $*" >&2; }
die()  { echo -e "${RED}error:${RESET} $*" >&2; exit 1; }

usage() { sed -n '2,/^set /p' "$0" | sed -n 's/^#\{1,\} \{0,1\}//p'; exit "${1:-0}"; }

# ── CUDA detection: pick the torch wheel stream from the installed driver ────
# nvidia-smi's header reports the MAX CUDA version the driver supports; torch
# wheels only need driver >= their build, so we take the newest known stream
# that the driver covers. Override with TORCH_CUDA=cuXXX (e.g. cu126) or
# TORCH_CUDA=cpu; no driver at all -> empty (PyPI's default torch build).
_TORCH_STREAMS="132 130 128 126 124 121 118"    # newest first; extend as PyTorch adds streams

driver_cuda() {   # -> e.g. "13.2", or "" if no usable driver
    # The || true guard matters: under set -e -o pipefail, a missing/failing
    # nvidia-smi would otherwise silently abort the whole script.
    { nvidia-smi 2>/dev/null || true; } | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1
}

detect_torch_index() {   # -> wheel index URL, or "" for PyPI default
    case "${TORCH_CUDA:-}" in
        cpu)   echo "https://download.pytorch.org/whl/cpu"; return ;;
        cu*)   echo "https://download.pytorch.org/whl/${TORCH_CUDA}"; return ;;
    esac
    local ver n s
    ver="$(driver_cuda)"
    [ -n "$ver" ] || return 0
    n=$(( ${ver%%.*} * 10 + ${ver#*.} ))        # "13.2" -> 132
    for s in $_TORCH_STREAMS; do
        if [ "$n" -ge "$s" ]; then echo "https://download.pytorch.org/whl/cu$s"; return; fi
    done
}

# ── steps ────────────────────────────────────────────────────────────────────

step_core() {
    local py="${PYTHON:-python3}"
    say "core: virtualenv vinkona_env (interpreter: $py — override with PYTHON=python3.13 if a dep lacks wheels for yours)"
    if [ ! -f vinkona_env/bin/activate ]; then
        rm -rf vinkona_env
        "$py" -m venv vinkona_env || die "venv creation failed — install python3-venv / python3-virtualenv first"
    fi
    say "core: python dependencies (requirements.txt — cascade only, no torch; TTS venvs and the legacy PersonaPlex stack carry their own)"
    ./vinkona_env/bin/pip install --upgrade pip -q
    local torch_idx; torch_idx="$(detect_torch_index)"
    if [ -n "$torch_idx" ]; then
        # No torch in core requirements — the extra index is a no-op unless a
        # transitive dep pulls torch, in which case it gets the right build.
        ./vinkona_env/bin/pip install -r requirements.txt --extra-index-url "$torch_idx"
    else
        ./vinkona_env/bin/pip install -r requirements.txt
    fi
    say "core: librnnoise (built and installed in-tree)"
    bash install_rnnoise.sh
    say "core: seeding live config (never overwrites an existing one)"
    mkdir -p config
    [ -f config/config.json ]   || cp config/config.example.json   config/config.json
    [ -f config/personas.json ] || cp config/personas.example.json config/personas.json
    ok "core installed — ASR models download into var/cache on first use"
}

step_tts() {
    local engine="${1:-orpheus}"
    case "$engine" in
        orpheus) bash install_orpheus.sh ;;
        neutts)  bash install_tts.sh ;;
        *) die "unknown TTS engine: $engine (orpheus|neutts)" ;;
    esac
}

_ggufs_in_models() { find -L Models -maxdepth 2 -name '*.gguf' 2>/dev/null | sort; }

_set_tier_model() {  # tier filename — update config/config.json (seed it first if absent)
    mkdir -p config
    [ -f config/config.json ] || cp config/config.example.json config/config.json
    python3 - "$1" "$2" <<'PY'
import json, sys
tier, fname = sys.argv[1], sys.argv[2]
path = "config/config.json"
cfg = json.load(open(path))
cfg.setdefault(tier, {})["model"] = fname
json.dump(cfg, open(path, "w"), indent=2)
print(f"  config: {tier}.model = {fname}")
PY
}

_assign_tiers() {
    local files=() f i c tier
    while IFS= read -r f; do [ -n "$f" ] && files+=("$f"); done <<<"$(_ggufs_in_models)"
    [ "${#files[@]}" -gt 0 ] || die "no .gguf files in Models/ — copy/symlink some in, or pick the download option"
    echo "Assign models to tiers (Enter keeps the current config value):"
    i=1; for f in "${files[@]}"; do echo "    $i) $(basename "$f")"; i=$((i+1)); done
    for tier in fast_lm big_lm embed_lm; do
        printf "  %s = 1-%d (Enter = keep current): " "$tier" "${#files[@]}"
        read -r c
        [ -z "$c" ] && continue
        case "$c" in *[!0-9]*) warn "not a number — keeping current $tier"; continue ;; esac
        if [ "$c" -ge 1 ] && [ "$c" -le "${#files[@]}" ]; then
            _set_tier_model "$tier" "$(basename "${files[$((c-1))]}")"
        else
            warn "out of range — keeping current $tier"
        fi
    done
    ok "tiers assigned — sanity-check with: ./serve_fast_lm.sh --dry-run"
    echo "  (embed_lm must be an embedding model, e.g. nomic-embed — a chat GGUF won't work there)"
}

step_models() {
    local existing tty=0 def c
    existing="$(_ggufs_in_models)"
    { [ -t 0 ] || [ "${VINKONA_ASSUME_TTY:-}" = 1 ]; } && tty=1
    if [ "$tty" -ne 1 ]; then
        # Non-interactive: deterministic — respect models that are already there.
        if [ -n "$existing" ]; then ok "models already present in Models/ — skipping download"; return 0; fi
        bash fetch_models.sh; return
    fi
    echo "Model weights — the LM tiers load GGUFs from Models/ (symlinks welcome)."
    if [ -n "$existing" ]; then
        echo "Already in Models/:"
        echo "$existing" | sed 's|^Models/|    |'
    fi
    echo "  1) download the default set from Hugging Face (fast 3B ≈2 GB, big 32B ≈20 GB, embed ≈260 MB)"
    echo "  2) use models already in Models/ (assign them to the fast/big/embed tiers)"
    echo "  3) link .gguf files from another folder into Models/, then assign tiers"
    echo "  s) skip for now"
    def=1; [ -n "$existing" ] && def=2
    printf "choice [%s]: " "$def"
    read -r c; c="${c:-$def}"
    case "$c" in
        1) bash fetch_models.sh ;;
        2) _assign_tiers ;;
        3) printf "folder containing your .gguf files: "
           read -r src; src="${src/#\~/$HOME}"
           [ -d "$src" ] || die "not a folder: $src"
           local n=0
           for f in "$src"/*.gguf; do
               [ -e "$f" ] || continue
               ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" Models/
               n=$((n+1))
           done
           [ "$n" -gt 0 ] || die "no .gguf files found in $src"
           ok "linked $n model file(s) into Models/"
           _assign_tiers ;;
        s|S) warn "skipped — run './install.sh models' again anytime" ;;
        *) die "unknown choice: $c" ;;
    esac
}

resolve_nvcc() {   # -> path to an actual CUDA compiler binary, or ""
    # Always returns 0: an empty result is data, not an error (set -e safety).
    if command -v nvcc >/dev/null 2>&1; then command -v nvcc; return 0; fi
    local c
    for c in /usr/local/cuda/bin/nvcc "${CUDA_HOME:-}/bin/nvcc" "${CUDA_PATH:-}/bin/nvcc"; do
        [ -x "$c" ] && { echo "$c"; return 0; }
    done
    return 0
}

_nvcc_probe() {   # nvcc [extra flags...] -> 0 if a trivial .cu compiles
    local nv="$1"; shift
    local t; t="$(mktemp -d "${TMPDIR:-/tmp}/nvcc-probe.XXXXXX")" || return 1
    echo 'int main(){return 0;}' > "$t/p.cu"
    local rc=0
    "$nv" "$@" -c "$t/p.cu" -o "$t/p.o" >/dev/null 2>&1 || rc=1
    rm -rf "$t"
    return $rc
}

step_llama() {
    if command -v llama-server >/dev/null 2>&1 && [ ! -x bin/llama-server ]; then
        ok "llama-server already on PATH: $(command -v llama-server) — skipping build (run './install.sh llama --force' to build in-tree anyway)"
        [ "${1:-}" = "--force" ] || return 0
    fi
    vk_require_tools git cmake gcc "g++:gcc-c++|g++" make \
        || die "building llama.cpp needs the C++ toolchain + cmake (see above)"

    # CUDA needs TWO things: the driver (nvidia-smi, to run) and the toolkit's
    # nvcc (to build). A bare /usr/local/cuda directory proves neither.
    local cuda_flag="-DGGML_CUDA=OFF" nvcc=""
    if [ -n "$(driver_cuda)" ]; then
        nvcc="$(resolve_nvcc)"
        if [ -z "$nvcc" ]; then
            say "llama: NVIDIA driver detected, but no CUDA compiler (nvcc) — required to BUILD GPU support"
            if vk_require_tools "nvcc:cuda-toolkit|nvidia-cuda-toolkit"; then
                nvcc="$(resolve_nvcc)"
            fi
        fi
        if [ -n "$nvcc" ]; then
            cuda_flag="-DGGML_CUDA=ON"
        else
            warn "no nvcc available — refusing to silently build a CPU-only llama-server for a GPU machine."
            warn "Your options:"
            warn "  1. Install the CUDA toolkit, then re-run this task"
            warn "     (Fedora: dnf install cuda-toolkit from NVIDIA's cuda repo — https://developer.nvidia.com/cuda-downloads;"
            warn "      Ubuntu/distrobox: apt install nvidia-cuda-toolkit)"
            warn "  2. Run this task INSIDE the CUDA distrobox (its image ships nvcc) — the LM services must then run there too"
            warn "  3. Already have a llama-server binary on this machine? Put it on PATH, symlink it into ./bin/, or set LLAMA_SERVER=/path"
            if [ -t 0 ] || [ "${VINKONA_ASSUME_TTY:-}" = 1 ]; then
                printf "Build CPU-ONLY anyway (slow for chat-size models)? [y/N]: "
                local a; read -r a
                case "$a" in y*|Y*) ;; *) die "aborted — GPU build needs nvcc (see options above)" ;; esac
            else
                die "aborted — GPU build needs nvcc (see options above)"
            fi
        fi
    fi

    # nvcc caps the host gcc major it accepts (e.g. CUDA 13.3 -> gcc <= 15) and
    # bleeding-edge distros ship newer. Probe; if the system compiler is
    # rejected, hunt for a compat g++-XX, else offer nvcc's documented
    # -allow-unsupported-compiler escape (explicit opt-in, never silent).
    local ccbin="" cuda_extra=""
    if [ "$cuda_flag" = "-DGGML_CUDA=ON" ] && ! _nvcc_probe "$nvcc"; then
        say "llama: nvcc rejects the system gcc (host compiler too new for this CUDA toolkit) — probing for a compatible one"
        local cand
        for cand in ${VINKONA_NVCC_CCBIN:-} g++-15 g++-14 g++-13 g++-12; do
            command -v "$cand" >/dev/null 2>&1 || continue
            if _nvcc_probe "$nvcc" -ccbin "$(command -v "$cand")"; then
                ccbin="$(command -v "$cand")"; break
            fi
        done
        if [ -n "$ccbin" ]; then
            ok "compatible host compiler: $ccbin"
        elif _nvcc_probe "$nvcc" -allow-unsupported-compiler; then
            warn "no CUDA-compatible host compiler found (checked g++-15..g++-12)."
            warn "  1. Install a compat gcc, then re-run this task"
            warn "     (Fedora: sudo dnf install gcc14-c++   Ubuntu: sudo apt install g++-14 — then it's picked up automatically)"
            warn "  2. Or force the build with nvcc's -allow-unsupported-compiler (works in practice for llama.cpp; NVIDIA says 'at your own risk')"
            if [ "${VINKONA_NVCC_ALLOW_UNSUPPORTED:-}" = 1 ]; then
                cuda_extra="-allow-unsupported-compiler"
            elif [ -t 0 ] || [ "${VINKONA_ASSUME_TTY:-}" = 1 ]; then
                printf "Force with -allow-unsupported-compiler? [Y/n]: "
                local a; read -r a
                case "$a" in n*|N*) die "aborted — install a compat gcc (option 1) and re-run" ;; esac
                cuda_extra="-allow-unsupported-compiler"
            else
                die "aborted — install a compat gcc (option 1), or set VINKONA_NVCC_ALLOW_UNSUPPORTED=1 to force"
            fi
        else
            die "nvcc can't compile even with -allow-unsupported-compiler — install a compat gcc (Fedora: gcc14-c++, Ubuntu: g++-14) and re-run"
        fi
    fi

    local src="$VINKONA_VAR/build/llama.cpp"
    say "llama: cloning/updating llama.cpp into $src"
    if [ -d "$src/.git" ]; then git -C "$src" pull --ff-only; else
        mkdir -p "$VINKONA_VAR/build"
        git clone --depth 1 https://github.com/ggml-org/llama.cpp "$src"
    fi
    say "llama: building llama-server ($cuda_flag${nvcc:+, nvcc: $nvcc}${ccbin:+, host: $ccbin}${cuda_extra:+, forced: $cuda_extra}) — this takes a while"
    cmake -S "$src" -B "$src/build" $cuda_flag ${nvcc:+-DCMAKE_CUDA_COMPILER="$nvcc"} \
          ${ccbin:+-DCMAKE_CUDA_HOST_COMPILER="$ccbin"} ${cuda_extra:+-DCMAKE_CUDA_FLAGS="$cuda_extra"} \
          -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON >/dev/null
    cmake --build "$src/build" --target llama-server -j"$(nproc)"
    mkdir -p bin
    cp "$src/build/bin/llama-server" bin/
    ok "installed bin/llama-server (env.sh puts ./bin on PATH for all Vinkona services)"
}

step_status() {
    echo "Vinkona assistant @ $SCRIPT_DIR"
    local cuda idx
    cuda="$(driver_cuda)"; idx="$(detect_torch_index)"
    echo "  gpu       driver CUDA: ${cuda:-none detected} → torch wheels: ${idx:-PyPI default} (override: TORCH_CUDA=cuXXX|cpu)"
    local d
    for d in vinkona_env orpheus_env neutts_env personaplex_env; do
        [ -d "$d" ] && echo "  venv      $d  ($(du -sh "$d" 2>/dev/null | cut -f1))"
    done
    [ -x bin/llama-server ] && echo "  binary    bin/llama-server" || {
        command -v llama-server >/dev/null 2>&1 && echo "  binary    llama-server (system PATH: $(command -v llama-server))" \
                                                || echo "  binary    llama-server MISSING — ./install.sh llama, or set LLAMA_SERVER"; }
    if [ -L Models ]; then echo "  models    Models -> $(readlink Models) (symlink; never touched by uninstall)"
    elif [ -d Models ]; then echo "  models    Models/ ($(du -sh Models 2>/dev/null | cut -f1), $(find Models -name '*.gguf' -o -name '*.safetensors' 2>/dev/null | wc -l) weight files)"; fi
    [ -d var ]    && echo "  caches    var/ ($(du -sh var 2>/dev/null | cut -f1))"
    [ -d logs ]   && echo "  runtime   logs/ ($(du -sh logs 2>/dev/null | cut -f1))"
    [ -f config/config.json ] && echo "  userdata  config/ (config.json, personas.json, memory.db, profiles — yours, kept on uninstall)"
    echo "  All of the above lives inside this folder — nothing is written elsewhere."
}

step_uninstall() {
    local with_models=0 purge=0 a
    for a in "$@"; do case "$a" in
        --with-models) with_models=1 ;;
        --purge)       purge=1 ;;
        *) die "unknown uninstall flag: $a" ;;
    esac; done

    say "removing generated artifacts (source files and user data are kept)"
    local d
    for d in vinkona_env orpheus_env neutts_env personaplex_env var bin moshi_src; do
        [ -e "$d" ] && { rm -rf "$d"; ok "removed $d/"; }
    done
    find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

    if [ "$with_models" -eq 1 ] || [ "$purge" -eq 1 ]; then
        if [ -L Models ]; then
            warn "Models is a symlink to $(readlink Models) — leaving the target alone, removing only the link"
            rm Models; mkdir Models; touch Models/.gitkeep
        elif [ -d Models ]; then
            rm -rf Models; mkdir Models; touch Models/.gitkeep; ok "removed downloaded weights in Models/"
        fi
    fi

    if [ "$purge" -eq 1 ]; then
        echo ""
        warn "--purge deletes YOUR DATA: config/ (memory.db, personas, live config, ws token) and logs/."
        warn "This is the assistant's memory of you. It cannot be recovered."
        printf "Type 'purge' to confirm: "
        read -r answer
        if [ "$answer" = "purge" ]; then
            # Only user data — the tracked *.example.json templates stay.
            rm -rf config/profiles logs certs
            rm -f config/config.json config/personas.json config/memory.db* \
                  config/active_profile config/trace.jsonl config/ws_token.txt
            ok "user data purged"
        else
            warn "skipped user-data purge"
        fi
    fi
    ok "uninstalled. What remains is the source tree (and your data unless purged) — delete the folder to remove everything."
}

# ── dispatch ─────────────────────────────────────────────────────────────────
cmd="${1:-core}"
case "$cmd" in
    core)       step_core ;;
    tts)        shift; step_tts "${1:-orpheus}" ;;
    models)     step_models ;;
    llama)      shift || true; step_llama "${1:-}" ;;
    all)        step_core; step_tts orpheus; step_models
                { command -v llama-server >/dev/null 2>&1 || [ -x bin/llama-server ]; } || step_llama ;;
    status)     step_status ;;
    uninstall)  shift || true; step_uninstall "$@" ;;
    -h|--help|help) usage 0 ;;
    *)          echo "unknown command: $cmd" >&2; usage 1 ;;
esac
