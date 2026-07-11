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

# ── steps ────────────────────────────────────────────────────────────────────

step_core() {
    say "core: virtualenv vinkona_env"
    if [ ! -f vinkona_env/bin/activate ]; then
        rm -rf vinkona_env
        python3 -m venv vinkona_env || die "venv creation failed — install python3-venv / python3-virtualenv first"
    fi
    say "core: python dependencies (requirements.txt — includes torch, first run is a big download)"
    ./vinkona_env/bin/pip install --upgrade pip -q
    ./vinkona_env/bin/pip install -r requirements.txt
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

step_models() { bash fetch_models.sh; }

step_llama() {
    if command -v llama-server >/dev/null 2>&1 && [ ! -x bin/llama-server ]; then
        ok "llama-server already on PATH: $(command -v llama-server) — skipping build (run './install.sh llama --force' to build in-tree anyway)"
        [ "${1:-}" = "--force" ] || return 0
    fi
    for tool in git cmake gcc g++ make; do
        command -v "$tool" >/dev/null 2>&1 || die "building llama.cpp needs $tool — install your distro's C++ toolchain + cmake"
    done
    local src="$VINKONA_VAR/build/llama.cpp"
    say "llama: cloning/updating llama.cpp into $src"
    if [ -d "$src/.git" ]; then git -C "$src" pull --ff-only; else
        mkdir -p "$VINKONA_VAR/build"
        git clone --depth 1 https://github.com/ggml-org/llama.cpp "$src"
    fi
    local cuda_flag="-DGGML_CUDA=OFF"
    if command -v nvcc >/dev/null 2>&1 || [ -d /usr/local/cuda ]; then cuda_flag="-DGGML_CUDA=ON"; fi
    say "llama: building llama-server ($cuda_flag) — this takes a while"
    cmake -S "$src" -B "$src/build" $cuda_flag -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON >/dev/null
    cmake --build "$src/build" --target llama-server -j"$(nproc)"
    mkdir -p bin
    cp "$src/build/bin/llama-server" bin/
    ok "installed bin/llama-server (env.sh puts ./bin on PATH for all Vinkona services)"
}

step_status() {
    echo "Vinkona assistant @ $SCRIPT_DIR"
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
