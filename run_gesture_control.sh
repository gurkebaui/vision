#!/usr/bin/env bash
set -e

echo "🚀 Vision Control v2 — Long-Range Gesture System"
echo "================================================="

PYTHON_BIN="python3"
if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif ! command -v python3 >/dev/null 2>&1; then
    echo "❌ Python 3.9+ not found. Install Python from https://python.org"
    exit 1
fi

if [ "$1" = "--studio" ] || [ "$1" = "--benchmark" ]; then
    exec "$PYTHON_BIN" gesture_presentation_control.py "$@"
fi

exec "$PYTHON_BIN" setup_and_run.py "$@"
