#!/bin/bash
# Mirror screening_cathode/, periodic_trend/, and README.md from batterymat_jae
# to batterymat whenever files change. fswatch (FSEvents) detects; rsync handles
# the two directories; an awk transform handles README.md (renames batterymat_jae
# -> batterymat and drops screening_alignn references during the copy).
#
# Usage:
#   sync_screening_to_batterymat.sh start    # start in background (idempotent)
#   sync_screening_to_batterymat.sh stop     # kill the background watcher
#   sync_screening_to_batterymat.sh status   # report running state
#   sync_screening_to_batterymat.sh run      # run watcher in foreground (used internally)
#   sync_screening_to_batterymat.sh sync     # one-shot mirror, no watcher

set -u

SRC_PKG="/Users/jaelee/Desktop/work/batterymat_jae/batterymat_jae"
DST_PKG="/Users/jaelee/Desktop/work/batterymat/batterymat"
SRC_README="/Users/jaelee/Desktop/work/batterymat_jae/README.md"
DST_README="/Users/jaelee/Desktop/work/batterymat/README.md"
LOG="/Users/jaelee/Desktop/work/.batterymat-sync.log"
PIDFILE="/Users/jaelee/Desktop/work/.batterymat-sync.pid"
FSWATCH="/opt/homebrew/bin/fswatch"

RSYNC_OPTS=(
  -a --delete
  --exclude '__pycache__/'
  --exclude '.DS_Store'
  --exclude 'results/'
)

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# Rewrite README on the fly:
#   - rename "batterymat_jae" -> "batterymat" everywhere
#   - drop the "## Related modules ..." section through the blank line(s)
#     before the next "## " heading
#   - drop the "└── screening_alignn/" tree line
#   - promote the "├── benchmarks/..." line above it back to "└── benchmarks/..."
transform_readme() {
  awk '
    { gsub(/batterymat_jae/, "batterymat") }
    /^## Related modules in this repo \(not part of the BatteryMat pipeline\)$/ {
      skip = 1
    }
    skip && /^## / && !/^## Related modules / {
      skip = 0
    }
    skip { next }
    /^└── screening_alignn\// { next }
    /^├── benchmarks\// && /Stage 5/ { sub(/^├── /, "└── ") }
    { print }
  ' "$SRC_README" > "$DST_README.tmp" && mv "$DST_README.tmp" "$DST_README"
}

sync_once() {
  rsync "${RSYNC_OPTS[@]}" "$SRC_PKG/screening_cathode/" "$DST_PKG/screening_cathode/"
  rsync "${RSYNC_OPTS[@]}" "$SRC_PKG/periodic_trend/"    "$DST_PKG/periodic_trend/"
  transform_readme
}

is_running() {
  [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

cmd_run() {
  echo "[$(ts)] watcher start (pid=$$); initial full sync ..." >> "$LOG"
  sync_once >> "$LOG" 2>&1
  echo "[$(ts)] initial sync done; entering watch loop" >> "$LOG"
  "$FSWATCH" --latency=1.0 -o \
      "$SRC_PKG/screening_cathode" \
      "$SRC_PKG/periodic_trend" \
      "$SRC_README" \
  | while read -r _; do
      {
        echo "[$(ts)] change detected -> rsync"
        sync_once
        echo "[$(ts)] rsync done"
      } >> "$LOG" 2>&1
    done
}

cmd_start() {
  if is_running; then
    echo "already running (pid=$(cat "$PIDFILE"))" >&2
    return 0
  fi
  nohup "$0" run >/dev/null 2>&1 &
  echo $! > "$PIDFILE"
  echo "started (pid=$(cat "$PIDFILE")); log: $LOG"
}

cmd_stop() {
  if is_running; then
    local pid; pid=$(cat "$PIDFILE")
    pkill -P "$pid" 2>/dev/null
    kill "$pid" 2>/dev/null
    rm -f "$PIDFILE"
    echo "stopped"
  else
    echo "not running" >&2
    rm -f "$PIDFILE"
  fi
}

cmd_status() {
  if is_running; then
    echo "running (pid=$(cat "$PIDFILE"))"
    echo "  log:    $LOG"
    echo "  source: $SRC_PKG/{screening_cathode,periodic_trend} + $SRC_README"
    echo "  dest:   $DST_PKG/{screening_cathode,periodic_trend} + $DST_README"
  else
    echo "not running"
  fi
}

case "${1:-start}" in
  start)  cmd_start ;;
  stop)   cmd_stop ;;
  status) cmd_status ;;
  run)    cmd_run ;;
  sync)   sync_once && echo "one-shot sync done" ;;
  *)      echo "usage: $0 {start|stop|status|run|sync}" >&2; exit 2 ;;
esac
