#!/usr/bin/env bash
# Start the camera alignment monitor in the background (idempotent).
# Usage: run.sh [DATA_DIR]   (default ~/camera_align; state only, nothing is written in the repo)
set -euo pipefail
DATA_DIR="${1:-$HOME/camera_align}"
mkdir -p "$DATA_DIR"
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="${PYTHON:-$HOME/dev/EgoVerse/.venv/bin/python}"
if [ -f "$DATA_DIR/monitor.pid" ] && kill -0 "$(cat "$DATA_DIR/monitor.pid")" 2>/dev/null; then
  echo "already running (pid $(cat "$DATA_DIR/monitor.pid")) at http://127.0.0.1:8090"; exit 0
fi
cd "$REPO"
PYTHONPATH="$REPO" nohup "$PYTHON" -u -m egomimic.robot.camera_align.monitor --data-dir "$DATA_DIR" >> "$DATA_DIR/monitor.log" 2>&1 &
echo $! > "$DATA_DIR/monitor.pid"
echo "started pid $(cat "$DATA_DIR/monitor.pid"); view through: ssh -L 8090:127.0.0.1:8090 rl2-yam  ->  http://localhost:8090"
