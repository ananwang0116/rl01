#!/bin/bash
# Register, list and stop the background jobs this project starts.
#
#   bash tools/proc.sh start <name> <logfile> -- <command...>   start as its own session, record identity
#   bash tools/proc.sh list                                      show every registered job and whether it is alive
#   bash tools/proc.sh stop <name>                               stop ONE verified job (INT, then TERM)
#   bash tools/proc.sh stop-all                                  stop every registered job
#
# Identity = process-group id + kernel start time + command line, stored in $PID_DIR/<name>.pid.
# A job is signalled only if all three still match, so a reused pid is never hit.
# Nothing is ever matched by name or pattern, and SIGKILL is never sent (stop reports it instead).
PID_DIR="${PID_DIR:-$(cd "$(dirname "$0")/.." && pwd)/runs/pids}"
mkdir -p "$PID_DIR"

starttime() { sed 's/^.*) //' "/proc/$1/stat" 2>/dev/null | awk '{print $20}'; }   # field 22 of /proc/pid/stat
alive_group() { [ -n "$1" ] && [ -n "$(ps -g "$1" -o pid= 2>/dev/null)" ]; }

# sets STATE to alive / stale / none, and PGID / START / CMD from the record.
# Never call it inside $(...): a subshell would throw the variables away.
state() {
  local f="$PID_DIR/$1.pid"
  PGID=""; START=""; CMD=""; STATE=none
  [ -f "$f" ] || return 0
  IFS=$'\t' read -r PGID START CMD < "$f"
  if [ -n "$PGID" ] && [ -d "/proc/$PGID" ] && [ "$(starttime "$PGID")" = "$START" ] \
     && [ "$(ps -o pgid= -p "$PGID" | tr -d ' ')" = "$PGID" ]; then
    STATE=alive
  else
    STATE=stale
  fi
}

cmd_start() {
  local name="$1" log="$2"; shift 2
  [ "$1" = "--" ] && shift
  [ -n "$name" ] && [ -n "$log" ] && [ $# -gt 0 ] || { echo "usage: proc.sh start <name> <logfile> -- <command...>"; exit 2; }
  state "$name"
  case "$STATE" in
    alive) echo "REFUSED: job '$name' is already running (pgid $PGID). Stop it first."; exit 1 ;;
    stale) echo "note: removing stale record for '$name' (pgid $PGID is gone or was reused)"; rm -f "$PID_DIR/$name.pid" ;;
  esac
  local raw="$PID_DIR/$name.raw"; rm -f "$raw"
  # The child writes its own pid before exec, so we never race on $!.
  # `setsid -f` forks by itself: a command put in the background with `&` from a non-interactive shell
  # inherits "SIGINT ignored", which made `kill -INT` useless on earlier training runs.
  setsid -f bash -c 'echo $$ > "$1"; shift; exec "$@"' _ "$raw" "$@" > "$log" 2>&1 < /dev/null
  for _ in $(seq 1 50); do [ -s "$raw" ] && break; sleep 0.1; done
  [ -s "$raw" ] || { echo "FAILED: child did not report its pid"; exit 1; }
  local pgid; pgid=$(cat "$raw"); rm -f "$raw"
  printf '%s\t%s\t%s\n' "$pgid" "$(starttime "$pgid")" "$*" > "$PID_DIR/$name.pid"
  echo "started '$name' pgid=$pgid log=$log"
}

cmd_list() {
  local any=0 f n
  for f in "$PID_DIR"/*.pid; do
    [ -e "$f" ] || continue
    any=1; n=$(basename "$f" .pid)
    state "$n"
    printf '%-24s %-6s pgid=%s  %s\n' "$n" "$STATE" "$PGID" "$(echo "$CMD" | cut -c1-80)"
  done
  [ $any = 1 ] || echo "(no registered jobs)"
}

wait_gone() { for _ in $(seq 1 "$2"); do alive_group "$1" || return 0; sleep 1; done; return 1; }

cmd_stop() {
  local name="$1"
  state "$name"
  case "$STATE" in
    none)  echo "'$name': no record"; return 0 ;;
    stale) echo "'$name': stale record (process gone or pid reused); removing it, signalling nothing"; rm -f "$PID_DIR/$name.pid"; return 0 ;;
  esac
  local pgid="$PGID"
  [ -n "$pgid" ] || { echo "'$name': record has no pgid; refusing to signal anything"; return 1; }
  echo "'$name': stopping pgid $pgid ($(echo "$CMD" | cut -c1-70))"
  kill -INT -- "-$pgid" 2>/dev/null
  if ! wait_gone "$pgid" 20; then
    echo "  still alive after SIGINT (it may be ignoring it: grep SigIgn /proc/$pgid/status); sending SIGTERM to the group"
    kill -TERM -- "-$pgid" 2>/dev/null
    wait_gone "$pgid" 15 || { echo "  STILL ALIVE after SIGTERM: not sending SIGKILL. Inspect: ps -g $pgid"; return 1; }
  fi
  rm -f "$PID_DIR/$name.pid"
  echo "  stopped; group $pgid has no processes left"
}

cmd_stop_all() {
  local f n rc=0
  for f in "$PID_DIR"/*.pid; do
    [ -e "$f" ] || continue
    cmd_stop "$(basename "$f" .pid)" || rc=1
  done
  return $rc
}

sub="$1"; shift
case "$sub" in
  start) cmd_start "$@" ;;
  list) cmd_list ;;
  stop) cmd_stop "$1" ;;
  stop-all) cmd_stop_all ;;
  *) sed -n '2,11p' "$0"; exit 2 ;;
esac
