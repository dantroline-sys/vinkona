#!/bin/bash
# Vinkona — top-level installer. Run this and pick off tasks until everything
# is green; each task delegates to the component installers, so re-running
# anything is safe and incremental.
#
#   ./install.sh                  # interactive: checklist -> pick a task -> repeat
#   ./install.sh status           # the checklist, nothing else
#   ./install.sh all              # run every missing task in order
#   ./install.sh <task>           # one task: assistant-core | tts [orpheus|neutts]
#                                 #   | models | llama | vinur
#   ./install.sh uninstall        # uninstall both components (keeps your data)
#                 --with-models   #   also delete downloaded weights
#                 --purge         #   ALSO delete user data + knowledge base (asks)
#
# The knowledge host is Vinur, its own repository since the 2026-07-13 split
# (https://github.com/dantroline-sys/vinur) — clone it next to this repo and
# the vinur task manages it. Found as: $VINUR_DIR > vinur_dir= in
# .vinkona-services > ../vinur > ./knowledge-host (legacy monorepo layout).
#
# Everything installed or written lives INSIDE this folder tree (and Vinur's
# inside its own) — see the "Filesystem guarantee" sections in the READMEs.
# Deleting the folders removes every trace.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VINUR="${VINUR_DIR:-}"
[ -n "$VINUR" ] || VINUR="$(sed -n 's/^vinur_dir=//p' "$ROOT/.vinkona-services" 2>/dev/null | head -1)" || true
case "$VINUR" in "~"*) VINUR="${HOME}${VINUR#\~}";; esac
[ -n "$VINUR" ] || { [ -d "$ROOT/../vinur" ] && VINUR="$(cd "$ROOT/../vinur" && pwd)"; } || true
[ -n "$VINUR" ] || VINUR="$ROOT/knowledge-host"

GREEN='\033[0;32m'; RED='\033[0;31m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
say() { echo -e "${CYAN}==>${RESET} $*"; }

usage() { sed -n '2,/^set /p' "$0" | sed -n 's/^#\{1,\} \{0,1\}//p'; exit "${1:-0}"; }

# ── task registry: id | description | check | action ────────────────────────
TASKS=(assistant-core tts models llama vinur)

desc() {
    case "$1" in
        assistant-core) echo "assistant core (vinkona_env: cascade, ASR, memory, research, config UI)" ;;
        tts)            echo "TTS engine (orpheus_gguf: llama.cpp + SNAC, no venv — recommended; or orpheus/vLLM, neutts)" ;;
        models)         echo "LM weights (download defaults from HF, or select models you copied in)" ;;
        llama)          echo "llama-server binary (llama.cpp — system PATH or built in-tree)" ;;
        vinur)          if [ -d "$VINUR" ]; then
                            echo "knowledge host — Vinur @ $VINUR (.venv + config; format flags for pdf/epub/zim)"
                        else
                            echo "knowledge host — no Vinur checkout; clone github.com/dantroline-sys/vinur alongside"
                        fi ;;
    esac
}

# A venv only counts as installed if its packages actually landed — a failed pip
# leaves bin/activate behind and a bare "does the venv exist" check then shows a
# green ✓ over an empty env (the ModuleNotFoundError trap). Probe site-packages
# on disk (no exec — works even for a venv built inside the container).
_venv_has() {  # venv-dir module-dir -> 0 if installed
    compgen -G "$1/lib*/python3.*/site-packages/$2" >/dev/null 2>&1
}

# The orpheus_gguf voice backbone: either an Orpheus-named GGUF sits in Models/,
# or the user picked/kept a differently-named one — then config tts_lm.model is
# the truth, so check that its file exists.
_orpheus_gguf_present() {
    [ -n "$(find -L assistant/Models -maxdepth 2 -iname '*orpheus*.gguf' -print -quit 2>/dev/null)" ] \
        && return 0
    python3 - 2>/dev/null <<'PY'
import json, sys
from pathlib import Path
try:
    cfg = json.load(open("assistant/config/config.json"))
except Exception:
    sys.exit(1)
m = (cfg.get("tts_lm") or {}).get("model")
if not m:
    sys.exit(1)
p = Path(m)
if not p.is_absolute():
    p = Path("assistant") / (cfg.get("models_dir") or "Models") / m
sys.exit(0 if p.exists() else 1)
PY
}

installed() {
    case "$1" in
        assistant-core) _venv_has assistant/vinkona_env faster_whisper ;;
        tts)            { _venv_has assistant/vinkona_env onnxruntime && _orpheus_gguf_present; } \
                        || _venv_has assistant/orpheus_env numpy || _venv_has assistant/neutts_env numpy ;;
        models)         [ -n "$(find -L assistant/Models -name '*.gguf' -print -quit 2>/dev/null)" ] ;;
        llama)          [ -x assistant/bin/llama-server ] || command -v llama-server >/dev/null 2>&1 ;;
        vinur)          [ -f "$VINUR/.venv/bin/activate" ] ;;   # installer smoke-tests itself
    esac
}

run_task() {
    local t="$1"; shift || true
    case "$t" in
        assistant-core) (cd assistant && ./install.sh core) ;;
        tts)
            # The TTS service runs INSIDE the distrobox at serve time, so install it
            # there when the container exists — a venv must be built with the python
            # that will run it (and for the vLLM engine that also sidesteps a too-new
            # host interpreter).  orpheus_gguf only adds a wheel + downloads, but the
            # wheel goes into vinkona_env, which lives in the container too.
            local box="${VINKONA_BOX:-vinkona-cuda}"
            if command -v distrobox >/dev/null 2>&1 && distrobox list 2>/dev/null | grep -qw "$box"; then
                say "installing TTS inside the container ($box) — where it runs"
                (cd assistant && distrobox enter "$box" -- ./install.sh tts "${1:-orpheus_gguf}")
            else
                say "no container '$box' found — installing TTS on this system (set VINKONA_BOX=name if yours differs)"
                (cd assistant && ./install.sh tts "${1:-orpheus_gguf}")
            fi ;;
        models)         (cd assistant && ./install.sh models) ;;
        llama)          (cd assistant && ./install.sh llama) ;;
        vinur)
            if [ ! -d "$VINUR" ]; then
                echo -e "${RED}no Vinur checkout found${RESET} — the knowledge host lives in its own repo now:"
                echo "  git clone https://github.com/dantroline-sys/vinur \"$ROOT/../vinur\""
                echo "  (or point VINUR_DIR at an existing checkout)"
                return 1
            fi
            (cd "$VINUR" && ./install.sh "$@") ;;
        *) echo "unknown task: $t" >&2; usage 1 ;;
    esac
}

checklist() {
    echo -e "${BOLD}Vinkona @ $ROOT${RESET}"
    local i=1 t mark
    for t in "${TASKS[@]}"; do
        if installed "$t"; then mark="${GREEN}✓${RESET}"; else mark="${RED}✗${RESET}"; fi
        echo -e "  $i. [$mark] $t — $(desc "$t")"
        i=$((i+1))
    done
    echo "  Everything installs inside this folder (Vinur inside its own); nothing is written elsewhere."
}

all_done() { local t; for t in "${TASKS[@]}"; do installed "$t" || return 1; done; }

# ── dispatch ─────────────────────────────────────────────────────────────────
cmd="${1:-}"
case "$cmd" in
    status)
        checklist ;;
    all)
        for t in "${TASKS[@]}"; do
            installed "$t" && { say "$t — already installed, skipping"; continue; }
            say "installing: $t"
            run_task "$t"
        done
        checklist ;;
    uninstall)
        shift || true
        say "uninstalling assistant"
        (cd assistant && ./install.sh uninstall "$@")
        if [ -d "$VINUR" ]; then
            say "uninstalling knowledge host (Vinur @ $VINUR)"
            if [ "${1:-}" = "--purge" ] || [ "${2:-}" = "--purge" ]; then
                (cd "$VINUR" && ./install.sh uninstall --purge)
            else
                (cd "$VINUR" && ./install.sh uninstall)
            fi
        else
            say "no Vinur checkout found — nothing to uninstall there"
        fi ;;
    -h|--help|help)
        usage 0 ;;
    "")
        # Interactive: checklist -> pick -> run -> repeat, until green or 'q'.
        if [ ! -t 0 ]; then checklist; echo "(non-interactive shell — use './install.sh all' or './install.sh <task>')"; exit 0; fi
        while true; do
            echo ""
            checklist
            if all_done; then
                echo ""
                say "everything is installed — start the stack with:  ./vinkona.sh start"
                break
            fi
            echo ""
            printf "Task number to run (a = all remaining, q = quit): "
            read -r choice
            case "$choice" in
                q|Q) break ;;
                a|A) exec "$0" all ;;
                ''|*[!0-9]*) echo "pick 1-${#TASKS[@]}, a, or q" ;;
                *)  if [ "$choice" -ge 1 ] && [ "$choice" -le "${#TASKS[@]}" ]; then
                        t="${TASKS[$((choice-1))]}"
                        extra=""
                        if [ "$t" = "tts" ]; then
                            printf "engine [orpheus_gguf] / orpheus (vLLM) / neutts: "; read -r extra
                        elif [ "$t" = "vinur" ]; then
                            printf "format flags (e.g. --all or --pdf --epub) [none]: "; read -r extra
                        fi
                        # shellcheck disable=SC2086
                        run_task "$t" $extra || echo -e "${RED}task failed${RESET} — fix the issue above and re-run it"
                    else
                        echo "pick 1-${#TASKS[@]}, a, or q"
                    fi ;;
            esac
        done ;;
    *)  # a named task, remaining args pass through (e.g. ./install.sh tts neutts)
        t="$cmd"; shift || true
        [ "$t" = "knowledge-host" ] && t=vinur      # pre-split name still works
        case " ${TASKS[*]} " in
            *" $t "*) run_task "$t" "$@"; echo ""; checklist ;;
            *) echo "unknown command or task: $t" >&2; usage 1 ;;
        esac ;;
esac
