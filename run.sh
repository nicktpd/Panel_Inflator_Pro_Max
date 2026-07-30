#!/usr/bin/env bash
# Panel Inflator Pro Max - daily driver.
# Starts the local server on 127.0.0.1:8177 and opens the app.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "[ERROR] No virtual environment found. Run ./setup.sh first." >&2
    exit 1
fi

exec .venv/bin/python -m backend.main
