#!/bin/bash
# Vinkona — top-level installer. Run this and pick off tasks until everything
# is green; each task delegates to the component installers, so re-running
# anything is safe and incremental.
#
#   ./install.sh                  # interactive: checklist -> pick a task -> repeat
#   ./install.sh status           # the checklist, nothing else
#   ./install.sh all              # run every missing task in order
#   ./install.sh <task>           # one task: assistant-core | tts [orpheus|neutts]
#                                 #   | models | llama | knowledge-host
#   ./install.sh uninstall        # uninstall both components (keeps your data)
#                 --with-models   #   also delete downloaded weights
#                 --purge         #   ALSO delete user data + knowledge base (asks)
#
# Everything installed or written lives INSIDE this folder tree — see the
# "Filesystem guarantee" sections in the READMEs. Deleting the folder removes
# every trace.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

GREEN='\033[0;32m'; RED='\033[0;31m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
say() { echo -e "${CYAN}==>${RESET} $*"; }

usage() { sed -n '2,/^set /p' "$0" | sed -n 's/^#\{1,\} \{0,1\}//p'; exit "${1:-0}"; }

# ── task registry: id | description | check | action ────────────────────────
TASKS=(assistant-core tts models llama knowledge-host)

desc() {
    case "$1" in
        assistant-core) echo "assistant core (vinkona_env: cascade, ASR, memory, research, config UI)" ;;
        tts)            echo "TTS engine (orpheus or neutts, own venv; needs CUDA for orpheus)" ;;
        models)         echo "LM weights (download defaults from HF, or select models you copied in)" ;;
        llama)          echo "llama-server binary (llama.cpp — system PATH or built in-tree)" ;;
        knowledge-host) echo "knowledge host (.venv + config; add format flags for pdf/epub/zim)" ;;
    esac
}

installed() {
    case "$1" in
        assistant-core) [ -f assistant/vinkona_env/bin/activate ] ;;
        tts)            [ -f assistant/orpheus_env/bin/activate ] || [ -f assistant/neutts_env/bin/activate ] ;;
        models)         [ -n "$(find -L assistant/Models -name '*.gguf' -print -quit 2>/dev/null)" ] ;;
        llama)          [ -x assistant/bin/llama-server ] || command -v llama-server >/dev/null 2>&1 ;;
        knowledge-host) [ -f knowledge-host/.venv/bin/activate ] ;;
    esac
}

run_task() {
    local t="$1"; shift || true
    case "$t" in
        assistant-core) (cd assistant && ./install.sh core) ;;
        tts)            (cd assistant && ./install.sh tts "${1:-orpheus}") ;;
        models)         (cd assistant && ./install.sh models) ;;
        llama)          (cd assistant && ./install.sh llama) ;;
        knowledge-host) (cd knowledge-host && ./install.sh "$@") ;;
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
    echo "  Everything installs inside this folder; nothing is written elsewhere."
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
        say "uninstalling knowledge-host"
        if [ "${1:-}" = "--purge" ] || [ "${2:-}" = "--purge" ]; then
            (cd knowledge-host && ./install.sh uninstall --purge)
        else
            (cd knowledge-host && ./install.sh uninstall)
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
                say "everything is installed — start the stack with:  cd assistant && ./vinkona.sh start"
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
                            printf "engine [orpheus]/neutts: "; read -r extra
                        elif [ "$t" = "knowledge-host" ]; then
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
        case " ${TASKS[*]} " in
            *" $t "*) run_task "$t" "$@"; echo ""; checklist ;;
            *) echo "unknown command or task: $t" >&2; usage 1 ;;
        esac ;;
esac
