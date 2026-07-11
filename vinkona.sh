#!/bin/bash
# Vinkona — start and stop the whole system from the repo root.
#
# Which services run on THIS machine is chosen on the first start — and
# remembered, if you like — so the knowledge host can live on a separate
# device from the assistant:
#
#   everything          assistant + knowledge host (one-machine setup)
#   assistant only      the knowledge host lives elsewhere
#   knowledge host only this device just serves the knowledge base
#
# Usage:
#   ./vinkona.sh start            # start what this machine runs (asks first time)
#   ./vinkona.sh stop             # stop it
#   ./vinkona.sh restart [svc]    # restart everything, or one assistant service
#   ./vinkona.sh status           # what's up
#   ./vinkona.sh attach           # attach the tmux session (Ctrl-b d to detach)
#   ./vinkona.sh services         # change what this machine runs / re-ask
#
# The assistant stack is managed by assistant/vinkona.sh (its own tmux session
# "vinkona"); the knowledge host runs in tmux session "vinkona-kb". The saved
# choice lives in .vinkona-services (machine-local, git-ignored).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source assistant/env.sh          # vk_require_tools; cache pinning is harmless here

CONF="$ROOT/.vinkona-services"
KB_SESSION="vinkona-kb"
CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'
say()  { echo -e "${CYAN}==>${RESET} $*"; }
warn() { echo -e "${YELLOW}warning:${RESET} $*" >&2; }

usage() { sed -n '2,/^set /p' "$0" | sed -n 's/^#\{1,\} \{0,1\}//p'; exit "${1:-0}"; }

# ── which services does this machine run? ────────────────────────────────────
SERVICES=""

choose_services() {
    echo "What should this machine run?"
    echo "  1) everything          — the assistant and the knowledge host (default)"
    echo "  2) assistant only      — the knowledge host lives on another device"
    echo "  3) knowledge host only — this device just serves the knowledge base"
    printf "choice [1]: "
    local c; read -r c
    case "${c:-1}" in
        1) SERVICES=both ;;
        2) SERVICES=assistant ;;
        3) SERVICES=knowledge-host ;;
        *) echo "pick 1, 2 or 3"; exit 1 ;;
    esac
    printf "Remember this choice? You can change it anytime with './vinkona.sh services'. [Y/n]: "
    local r; read -r r
    case "$r" in
        n*|N*) rm -f "$CONF"; echo "Okay — I'll ask again next start." ;;
        *)     printf 'services=%s\n' "$SERVICES" > "$CONF"
               echo "Saved to .vinkona-services." ;;
    esac
}

resolve_services() {
    if [ -f "$CONF" ]; then
        SERVICES="$(sed -n 's/^services=//p' "$CONF" | head -1)"
        case "$SERVICES" in both|assistant|knowledge-host) return 0 ;; esac
        warn ".vinkona-services is malformed — re-asking"
    fi
    if [ -t 0 ] || [ "${VINKONA_ASSUME_TTY:-}" = 1 ]; then
        choose_services
    else
        SERVICES=both
        echo "(no saved service selection — defaulting to everything; set one with './vinkona.sh services')"
    fi
}

runs_assistant() { [ "$SERVICES" = both ] || [ "$SERVICES" = assistant ]; }
runs_kb()        { [ "$SERVICES" = both ] || [ "$SERVICES" = knowledge-host ]; }

# ── knowledge host (tmux session vinkona-kb) ─────────────────────────────────
kb_port() {
    local p
    p="$(sed -n 's/^port *= *\([0-9][0-9]*\).*/\1/p' knowledge-host/config.toml 2>/dev/null | head -1)"
    echo "${p:-8771}"
}

kb_start() {
    vk_require_tools tmux || exit 1
    if tmux has-session -t "$KB_SESSION" 2>/dev/null; then
        say "knowledge host: already running (tmux session $KB_SESSION)"
        return 0
    fi
    if [ ! -f knowledge-host/config.toml ] && [ ! -x knowledge-host/.venv/bin/python3 ]; then
        warn "the knowledge host doesn't look installed here — run ./install.sh first"
        return 1
    fi
    say "knowledge host: starting (tmux session $KB_SESSION, port $(kb_port))"
    tmux new-session -d -s "$KB_SESSION" -c "$ROOT/knowledge-host" \
        "bash -c './run.sh 2>&1 | tee -a var/service.log'"
}

kb_stop() {
    if tmux kill-session -t "$KB_SESSION" 2>/dev/null; then
        say "knowledge host: stopped"
    else
        say "knowledge host: not running"
    fi
}

kb_status() {
    if tmux has-session -t "$KB_SESSION" 2>/dev/null; then
        local port; port="$(kb_port)"
        local health=""
        command -v curl >/dev/null 2>&1 && health="$(curl -s -m 2 "localhost:$port/health" 2>/dev/null || true)"
        if [ -n "$health" ]; then
            echo -e "  knowledge host  ${GREEN}up${RESET}  (tmux $KB_SESSION, :$port — health ok)"
        else
            echo -e "  knowledge host  ${YELLOW}session up, service not answering yet${RESET}  (tmux attach -t $KB_SESSION to look)"
        fi
    else
        echo "  knowledge host  down"
    fi
}

# ── dispatch ─────────────────────────────────────────────────────────────────
cmd="${1:-}"; shift || true
case "$cmd" in
    start)
        resolve_services
        runs_kb        && kb_start
        runs_assistant && (cd assistant && ./vinkona.sh start "$@")
        say "done — './vinkona.sh status' to check, './vinkona.sh attach' to watch"
        ;;
    stop)
        resolve_services
        runs_assistant && (cd assistant && ./vinkona.sh stop "$@")
        runs_kb        && kb_stop
        ;;
    restart)
        resolve_services
        if runs_assistant && [ $# -gt 0 ]; then
            # restart one assistant service, e.g. ./vinkona.sh restart cascade
            (cd assistant && ./vinkona.sh restart "$@")
        else
            runs_assistant && (cd assistant && ./vinkona.sh restart)
            if runs_kb; then kb_stop; kb_start; fi
        fi
        ;;
    status)
        resolve_services
        echo "Vinkona @ $ROOT  (this machine runs: $SERVICES)"
        runs_assistant && (cd assistant && ./vinkona.sh status "$@") || true
        runs_kb        && kb_status
        ;;
    attach)
        resolve_services
        if runs_assistant && tmux has-session -t vinkona 2>/dev/null; then
            (cd assistant && ./vinkona.sh attach)
        elif tmux has-session -t "$KB_SESSION" 2>/dev/null; then
            runs_assistant && say "the assistant session isn't running (./vinkona.sh start) — attaching the knowledge host instead"
            exec tmux attach -t "$KB_SESSION"
        else
            echo "nothing is running — './vinkona.sh start' first"
            exit 1
        fi
        ;;
    services)
        if [ -t 0 ] || [ "${VINKONA_ASSUME_TTY:-}" = 1 ]; then
            choose_services
        else
            echo "'services' is interactive — or write .vinkona-services yourself:"
            echo "  echo 'services=knowledge-host' > .vinkona-services   # both | assistant | knowledge-host"
            exit 1
        fi
        ;;
    -h|--help|help) usage 0 ;;
    "")             usage 0 ;;
    *) echo "unknown command: $cmd" >&2; usage 1 ;;
esac
