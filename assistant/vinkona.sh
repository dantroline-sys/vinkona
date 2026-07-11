#!/bin/bash
# One command to run the whole Vinkona stack in a tmux session.
#
# Split: the LM services (llama.cpp) run on the Fedora HOST; the Python services +
# TTS run inside the distrobox container.  This script (run ON THE HOST) launches
# host services directly and container services via `distrobox enter`, one tmux
# window each.  Host and container share the network (localhost), so the ports line
# up across both.
#
# Each service's output is also written to logs/<name>.log (shared filesystem), so
# the config web UI can tail it and trigger restarts (a "monitor" window watches
# logs/control/ for restart requests written by the web UI).
#
#   ./vinkona.sh start            # bring everything up in tmux session "vinkona"
#   ./vinkona.sh attach            # attach (detach again with Ctrl-b then d)
#   ./vinkona.sh stop              # stop everything
#   ./vinkona.sh restart           # stop + start (everything)
#   ./vinkona.sh restart cascade   # restart just one service
#   ./vinkona.sh status            # list windows
#   ./vinkona.sh plan              # print what each window would run (no side effects)
#
# Set your distrobox container name if it isn't "vinkona-cuda":  VINKONA_BOX=name ./vinkona.sh start
set -u
SESSION="vinkona"
BOX="${VINKONA_BOX:-vinkona-cuda}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$DIR/logs"

# name | where (host|box) | command | kill-pattern (box only, to reap orphans on restart)
# The tts pattern must also match vLLM's detached children — Orpheus runs on vLLM,
# whose v1 engine forks a separate "VLLM::EngineCore" (+ "VLLM::Worker") process that
# holds the GPU memory and survives a kill of tts_server.py.  If it isn't reaped, the
# relaunched TTS can't allocate VRAM and silently fails.  reap_box() also KILLs and
# waits for the GPU to free, so the patterns just need to find every piece.
# Stack mode (normal | knowledge), persisted in a one-line control file the web UI can write.
#   normal     — the full live stack (voice path + Vinkona's own learning).
#   knowledge  — knowledge-acquisition: live voice path DOWN, TWO big LMs (3090 + 4090) + embed
#                serve the knowledge-host so it can split distillation across both (~2x). The
#                config UI stays up; flip back to normal when the import is done.
MODE_FILE="$LOGS/control/mode"
read_mode() {
  local m=normal
  [ -f "$MODE_FILE" ] && m="$(tr -d '[:space:]' < "$MODE_FILE" 2>/dev/null)"
  case "$m" in knowledge) echo knowledge;; *) echo normal;; esac
}

# Memory watchdog: llama.cpp's embedding server slowly leaks under heavy use (e.g. a
# knowledge-host import) until it OOMs and dies — and nothing brings it back.  The watchdog
# pre-empts that: it checks each watched LM's resident memory and restarts it (via the SAME
# logs/control path the web UI uses) when it crosses an RSS cap OR has died.  Disable with
# VINKONA_WATCHDOG=0.   VINKONA_WATCH entries are  name:port:rss_cap_MB  (cap 0 = crash-only).
WATCH_SPECS="${VINKONA_WATCH:-embed:11437:6000}"
WATCH_INTERVAL="${VINKONA_WATCH_INTERVAL:-20}"

set_services() {                           # populate SERVICES for the current mode
  if [ "$(read_mode)" = knowledge ]; then
    SERVICES=(
      "big_lm|host|./serve_big_lm.sh|"
      "big_lm2|host|./serve_big_lm2.sh|"
      "embed|host|./serve_embed.sh|"
      "config|box|./serve_config.sh|config_server.py"
    )
  else
    SERVICES=(
      "fast_lm|host|./serve_fast_lm.sh|"
      "big_lm|host|./serve_big_lm.sh|"
      "embed|host|./serve_embed.sh|"
      "tunnel|host|./serve_tunnel.sh|"
      "tts|box|./serve_tts.sh orpheus|tts_server\.py|VLLM::|EngineCore"
      "cascade|box|./serve_cascade.sh|cascade_server.py"
      "config|box|./serve_config.sh|config_server.py"
      "research|box|./serve_research.sh|research_worker.py"
    )
  fi
}

reap_box() {                               # pattern -> thorough kill INSIDE the box
  # SIGTERM, wait for processes (incl. detached vLLM workers) to exit and release the
  # GPU, then SIGKILL any stragglers.  Run for box services so a restart can't collide
  # with an orphan still holding VRAM.
  #
  # The pattern travels as an ENV VAR, not interpolated into the script text:
  # otherwise the reaper shell's own command line contains the pattern, pkill -f
  # matches it, and the reaper kills itself mid-job (bash then vomits the whole
  # multi-line command as a 'Terminated' notice).
  local pat="$1"
  distrobox enter "$BOX" -- env VK_REAP_PAT="$pat" bash -lc '
    pkill -TERM -f "$VK_REAP_PAT" 2>/dev/null
    for i in 1 2 3 4 5 6 7 8; do pgrep -f "$VK_REAP_PAT" >/dev/null 2>&1 || break; sleep 1; done
    pkill -KILL -f "$VK_REAP_PAT" 2>/dev/null
    true
  ' 2>/dev/null
}

pane_cmd() {                               # name where command... -> the shell line for the pane
  local name="$1" where="$2"; shift 2
  local log="$LOGS/$name.log"
  local inner="cd $(printf %q "$DIR") && $* 2>&1 | tee $(printf %q "$log")"
  if [ "$where" = "box" ]; then
    printf 'distrobox enter %q -- bash -lc %q' "$BOX" "$inner"
  else
    printf '%s' "$inner"
  fi
}

launch_window() {                          # name where command...
  local name="$1" where="$2"; shift 2
  tmux new-window -t "$SESSION" -n "$name"
  tmux send-keys -t "$SESSION:$name" "$(pane_cmd "$name" "$where" "$@")" C-m
}

restart_one() {
  local target="$1" s name where cmd killpat
  for s in "${SERVICES[@]}"; do
    IFS='|' read -r name where cmd killpat <<<"$s"
    [ "$name" = "$target" ] || continue
    echo "restarting $name"
    tmux kill-window -t "$SESSION:$name" 2>/dev/null
    if [ "$where" = "box" ] && [ -n "$killpat" ]; then
      reap_box "$killpat"                  # reap orphans (incl. vLLM EngineCore) + free VRAM
    fi
    sleep 1
    # shellcheck disable=SC2086
    launch_window "$name" "$where" $cmd
    return 0
  done
  echo "unknown service: $target  (known: $(for s in "${SERVICES[@]}"; do echo -n "${s%%|*} "; done))"
  return 1
}

start() {
  command -v tmux >/dev/null || { echo "tmux is not installed (host)."; exit 1; }
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session '$SESSION' is already running — use './vinkona.sh restart' or 'attach'."; exit 0
  fi
  mkdir -p "$LOGS/control"
  # Wake the container ONCE, alone, before the box windows launch: firing four
  # simultaneous `distrobox enter` calls at a stopped container races its cold
  # boot, and whichever service loses (often the cascade) dies on first start.
  if printf '%s\n' "${SERVICES[@]}" | grep -q '|box|'; then
    echo "waking the container ($BOX) ..."
    distrobox enter "$BOX" -- true 2>/dev/null \
      || echo "warning: couldn't enter container '$BOX' — box services will fail (create it, or VINKONA_BOX=name ./vinkona.sh start)"
  fi
  local first=1 s name where cmd killpat
  for s in "${SERVICES[@]}"; do
    IFS='|' read -r name where cmd killpat <<<"$s"
    if [ "$first" = 1 ]; then
      tmux new-session -d -s "$SESSION" -n "$name"
      tmux send-keys -t "$SESSION:$name" "$(pane_cmd "$name" "$where" $cmd)" C-m
      first=0
    else
      # shellcheck disable=SC2086
      launch_window "$name" "$where" $cmd
    fi
    [ "$where" = "host" ] && sleep 1       # let the LM servers start loading first
  done
  # control window: processes restart requests from the web UI (host-side, has tmux)
  tmux new-window -t "$SESSION" -n "monitor"
  tmux send-keys -t "$SESSION:monitor" "$(printf 'cd %q && VINKONA_BOX=%q ./vinkona.sh _monitor' "$DIR" "$BOX")" C-m
  # watchdog window: revive / pre-empt-OOM the embed LM (and any VINKONA_WATCH entry)
  local extra="monitor"
  if [ "${VINKONA_WATCHDOG:-1}" != 0 ]; then
    tmux new-window -t "$SESSION" -n "watchdog"
    tmux send-keys -t "$SESSION:watchdog" "$(printf 'cd %q && VINKONA_WATCH=%q VINKONA_WATCH_INTERVAL=%q ./vinkona.sh _watchdog' "$DIR" "$WATCH_SPECS" "$WATCH_INTERVAL")" C-m
    extra="monitor + watchdog"
  fi
  echo "started '$SESSION' (${#SERVICES[@]} services + $extra).  Attach: ./vinkona.sh attach"
}

stop() {
  tmux kill-session -t "$SESSION" 2>/dev/null && echo "killed tmux session '$SESSION'"
  pkill -f 'llm_server\.py|llama-server|serve_tunnel\.sh|8765:127\.0\.0\.1:8765' 2>/dev/null
  # Reap the box services + vLLM's detached EngineCore/Worker so no GPU memory leaks.
  reap_box 'tts_server\.py|cascade_server\.py|config_server\.py|research_worker\.py|VLLM::|EngineCore'
  # If anything above died holding the pty in a raw state (the old reaper
  # self-kill did exactly this), put the terminal back so typing stays visible.
  [ -t 0 ] && stty sane 2>/dev/null
  echo "stopped."
}

monitor() {                                # internal: watch for restart requests from the UI
  local cdir="$LOGS/control" f svc
  mkdir -p "$cdir"
  echo "[monitor] watching $cdir for restart requests (written by the config web UI)"
  while true; do
    for f in "$cdir"/*.req; do
      [ -e "$f" ] || continue
      svc="$(basename "$f" .req)"; rm -f "$f"
      if [ "$svc" = "__restart__" ]; then
        echo "[monitor] full restart (mode → $(read_mode)) requested"
        # A full restart kills this very session (and the monitor), so run it DETACHED.
        setsid bash -c "cd $(printf %q "$DIR") && VINKONA_BOX=$(printf %q "$BOX") ./vinkona.sh restart" \
          >>"$LOGS/_restart.log" 2>&1 </dev/null &
        disown 2>/dev/null || true
        continue
      fi
      echo "[monitor] restart request: $svc"
      restart_one "$svc"
    done
    sleep 1
  done
}

request_restart() {                        # name reason -> rate-limited restart request
  local name="$1" reason="$2" cdir="$LOGS/control" tsf now last cool
  tsf="$cdir/.$name.wd"; cool=$(( WATCH_INTERVAL * 4 ))     # don't re-request while restarting
  now="$(date +%s)"; last="$(cat "$tsf" 2>/dev/null || echo 0)"
  [ $(( now - last )) -ge "$cool" ] || return 0
  echo "$now" > "$tsf"
  echo "[watchdog] $(date '+%H:%M:%S') $name: $reason -> restart"
  : > "$cdir/$name.req"                     # the monitor consumes this and restarts cleanly
}

watchdog() {                               # internal: pre-empt the embed leak / revive a dead LM
  mkdir -p "$LOGS/control"
  echo "[watchdog] watching '$WATCH_SPECS' every ${WATCH_INTERVAL}s (cap = RSS MB; 0 = crash-only)"
  local spec name port cap pids p r rss
  while true; do
    for spec in $WATCH_SPECS; do
      IFS=':' read -r name port cap <<<"$spec"
      pids="$(pgrep -f -- "--port $port" 2>/dev/null)"
      if [ -z "$pids" ]; then              # the LM is down — revive it, but only if the stack
        tmux has-session -t "$SESSION" 2>/dev/null || continue          # is actually up and
        tmux list-windows -t "$SESSION" -F '#W' 2>/dev/null | grep -qx "$name" || continue  # owns it
        request_restart "$name" "not running"
        continue
      fi
      [ "${cap:-0}" -gt 0 ] 2>/dev/null || continue        # cap 0 ⇒ crash-recovery only
      rss=0
      for p in $pids; do
        r="$(ps -o rss= -p "$p" 2>/dev/null | tr -d ' ')"
        [ -n "$r" ] && rss=$(( rss + r / 1024 ))           # KB -> MB
      done
      [ "$rss" -gt "$cap" ] && request_restart "$name" "RSS ${rss}MB > ${cap}MB"
    done
    sleep "$WATCH_INTERVAL"
  done
}

mkdir -p "$LOGS/control"
case "${1:-}" in
  start)    [ -n "${2:-}" ] && printf '%s\n' "$2" > "$MODE_FILE"     # ./vinkona.sh start knowledge
            set_services; start ;;
  stop)     set_services; stop ;;
  restart)  if [ "${2:-}" = normal ] || [ "${2:-}" = knowledge ]; then
              printf '%s\n' "$2" > "$MODE_FILE"; set_services; stop; sleep 2; start   # mode switch
            elif [ -n "${2:-}" ]; then set_services; restart_one "$2"                 # one service
            else set_services; stop; sleep 2; start; fi ;;
  attach)   tmux attach -t "$SESSION" ;;
  status)   tmux list-windows -t "$SESSION" 2>/dev/null || echo "session '$SESSION' not running" ;;
  mode)     read_mode ;;
  plan)     set_services; for s in "${SERVICES[@]}"; do IFS='|' read -r n w c k <<<"$s"
              printf '%-9s [%s]  %s\n' "$n" "$w" "$(pane_cmd "$n" "$w" $c)"; done ;;
  _monitor) set_services; monitor ;;
  _watchdog) watchdog ;;
  *) echo "usage: $0 {start [mode]|stop|restart [svc|normal|knowledge]|attach|status|mode|plan}"
     echo "       mode = normal | knowledge   (current: $(read_mode), BOX=$BOX)"
     echo "       watchdog: VINKONA_WATCHDOG=0 disables; VINKONA_WATCH='embed:11437:6000' name:port:cap_MB"; exit 1 ;;
esac
