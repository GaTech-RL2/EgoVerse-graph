#!/usr/bin/env bash
# Stop the camera alignment monitor started by run.sh. Usage: stop.sh [DATA_DIR]
DATA_DIR="${1:-$HOME/camera_align}"
if [ -f "$DATA_DIR/monitor.pid" ]; then
  kill "$(cat "$DATA_DIR/monitor.pid")" 2>/dev/null && echo "stopped pid $(cat "$DATA_DIR/monitor.pid")"
  rm -f "$DATA_DIR/monitor.pid"
else
  echo "not running"
fi
