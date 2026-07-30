#!/usr/bin/env bash
# Panel Inflator Pro Max - one-time setup (per machine).
# Creates a local .venv and installs Python dependencies.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    for c in python3.13 python3.12 python3.11 python3; do
        if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
    done
fi
if [ -z "$PY" ]; then
    echo "[ERROR] Python 3.11+ not found. Install it first." >&2
    exit 1
fi

echo "Creating virtual environment in .venv using $PY ..."
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Setup complete. Run ./run.sh to start Panel Inflator Pro Max."
echo "Remember: exclude .venv from cloud sync (see SYNC-IGNORE.txt)."
