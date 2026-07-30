#!/usr/bin/env bash
# Panel Inflator Pro Max - manual update.
# Pulls the latest version and refreshes dependencies. run.sh also does
# this automatically at launch; use this to force it or see errors.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .git ]; then
    echo "[ERROR] Not a git install; re-download instead." >&2
    exit 1
fi

echo "Pulling latest version..."
git pull --ff-only

if [ -x ".venv/bin/python" ]; then
    echo "Refreshing dependencies..."
    .venv/bin/python -m pip install -q -r requirements.txt
fi

echo "Up to date. Run ./run.sh to start the app."
