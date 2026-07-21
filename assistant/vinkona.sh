#!/bin/bash
# One command to run the whole Vinkona stack — a thin shim over supervisor.py,
# the stdlib-Python process supervisor that owns every service (see its header
# for how placement, restarts, the watchdog and the web-UI control files work).
#
#   ./vinkona.sh setup            # first run: install + models + start, NO questions
#   ./vinkona.sh doctor           # is this machine ready? what one command fixes it?
#   ./vinkona.sh start            # bring everything up under the supervisor
#   ./vinkona.sh stop             # stop everything
#   ./vinkona.sh restart          # restart everything
#   ./vinkona.sh restart cascade  # restart just one service
#   ./vinkona.sh status           # supervisor + per-service health
#   ./vinkona.sh logs [svc]       # follow one log, or all multiplexed
#   ./vinkona.sh plan             # print what would run (no side effects)
#
# Stack mode:  ./vinkona.sh start knowledge   |   ./vinkona.sh restart normal
#   normal     — the full live stack (voice path + Vinkona's own learning).
#   knowledge  — knowledge-acquisition: live voice path DOWN, TWO big LMs +
#                embed serve the knowledge host (~2x distillation); config UI
#                stays up.  Flip back with 'restart normal' when done.
#
# Set your distrobox container name if it isn't "vinkona-cuda":
#   VINKONA_BOX=name ./vinkona.sh start     (no container at all -> host-only)
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cmd="${1:-}"
case "$cmd" in
    setup)   # the LM-Studio path: one command from clone to chatting, zero questions.
        # stdin is redirected so install.sh takes its deterministic non-interactive
        # branches (prebuilt llama-server, RAM-sized model set, seeded config).
        cd "$DIR"
        echo "Vinkona setup — install, models, start.  No questions; sensible defaults."
        bash install.sh all </dev/null || { echo; python3 doctor.py; exit 1; }
        python3 supervisor.py start || exit 1
        echo
        python3 doctor.py
        exit 0 ;;
    doctor)
        exec python3 "$DIR/doctor.py" ;;
    attach)   # tmux is gone — the supervisor writes logs/<name>.log instead.
        echo "(the stack runs under supervisor.py now — following logs; Ctrl-C detaches)"
        shift
        exec python3 "$DIR/supervisor.py" logs "$@" ;;
    start|stop|restart|status|plan|mode|logs)
        exec python3 "$DIR/supervisor.py" "$@" ;;
    -h|--help|help|"")
        sed -n '2,/^set /p' "$0" | sed -n 's/^#\{1,\} \{0,1\}//p'; exit 0 ;;
    *)
        echo "unknown command: $cmd" >&2
        sed -n '2,/^set /p' "$0" | sed -n 's/^#\{1,\} \{0,1\}//p'; exit 1 ;;
esac
