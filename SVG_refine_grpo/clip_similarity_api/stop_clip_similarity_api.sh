#!/usr/bin/env bash
set -euo pipefail

PID_FILE="$(cd "$(dirname "$0")" && pwd)/server.pid"
if [ ! -f "$PID_FILE" ]; then
  echo "[WARN] server.pid not found"
  exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "[INFO] stopped pid=$PID"
else
  echo "[WARN] pid=$PID not running"
fi
rm -f "$PID_FILE"
