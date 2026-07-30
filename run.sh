#!/usr/bin/env bash
# Panel Inflator Pro Max - daily driver.
# Auto-updates from GitHub (safe no-op when offline), then starts the
# local server on 127.0.0.1:8177 and opens the app.
# Set PIPM_NO_UPDATE=1 to launch without checking for updates.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "[ERROR] No virtual environment found. Run ./setup.sh first." >&2
    exit 1
fi

.venv/bin/python -m backend.autoupdate || true
exec .venv/bin/python -m backend.main
