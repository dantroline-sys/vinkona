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
#   ./vinkona.sh restart [svc]    # restart everything, or one service (incl. kb)
#   ./vinkona.sh status           # what's up
#   ./vinkona.sh logs [svc]       # follow service logs (Ctrl-C detaches)
#   ./vinkona.sh services         # change what this machine runs / re-ask
#
# Everything runs under one process supervisor (assistant/supervisor.py) —
# the knowledge host is just another service there, so status/restart/logs
# cover it too.  The saved choice lives in .vinkona-services (git-ignored).
#
# The knowledge host is Vinur, its own repository since the 2026-07-13 split
# (https://github.com/dantroline-sys/vinur). This script finds the checkout as:
#   $VINUR_DIR > vinur_dir= in .vinkona-services > ../vinur (sibling clone)
#   > ./knowledge-host (legacy monorepo layout)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONF="$ROOT/.vinkona-services"
# The Vinur checkout (see header). Resolved once.
VINUR="${VINUR_DIR:-}"
[ -n "$VINUR" ] || VINUR="$(sed -n 's/^vinur_dir=//p' "$CONF" 2>/dev/null | head -1)"
case "$VINUR" in "~"*) VINUR="${HOME}${VINUR#\~}";; esac
[ -n "$VINUR" ] || { [ -d "$ROOT/../vinur" ] && VINUR="$(cd "$ROOT/../vinur" && pwd)"; }
[ -n "$VINUR" ] || VINUR="$ROOT/knowledge-host"

CYAN='\033[0;36m'; YELLOW='\033[1;33m'; RESET='\033[0m'
say()  { echo -e "${CYAN}==>${RESET} $*"; }
warn() { echo -e "${YELLOW}warning:${RESET} $*" >&2; }

usage() { sed -n '2,/^set /p' "$0" | sed -n 's/^#\{1,\} \{0,1\}//p'; exit "${1:-0}"; }

sup() { python3 "$ROOT/assistant/supervisor.py" "$@"; }

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

runs_kb() { [ "$SERVICES" = both ] || [ "$SERVICES" = knowledge-host ]; }

kb_flags() {   # -> the supervisor start flags for this machine's selection
    if ! runs_kb; then
        echo "--assistant-only"
        return
    fi
    if [ ! -d "$VINUR" ]; then
        warn "no Vinur checkout found — clone https://github.com/dantroline-sys/vinur"
        warn "next to this repo (../vinur), or point VINUR_DIR at it"
        echo "--assistant-only"
        return
    fi
    if [ "$SERVICES" = knowledge-host ]; then
        echo "--kb $VINUR --kb-only"
    else
        echo "--kb $VINUR"
    fi
}

# ── dispatch ─────────────────────────────────────────────────────────────────
cmd="${1:-}"; shift || true
case "$cmd" in
    start)
        resolve_services
        # shellcheck disable=SC2046
        sup start "$@" $(kb_flags)
        say "done — './vinkona.sh status' to check, './vinkona.sh logs' to watch"
        ;;
    stop)     sup stop ;;
    restart)  sup restart "$@" ;;
    status)
        resolve_services
        echo "Vinkona @ $ROOT  (this machine runs: $SERVICES)"
        sup status
        ;;
    logs)     sup logs "$@" ;;
    attach)   echo "(tmux is gone — the supervisor writes logs/<name>.log; following)"
              sup logs "$@" ;;
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
