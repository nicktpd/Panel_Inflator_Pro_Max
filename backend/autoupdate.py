"""Best-effort self-update, run by run.bat / run.sh before the server starts.

Workflow this enables: ask Claude (in any session) to change the app and
push to main; the next time you double-click run.bat you are on the
latest version. No manual pulling, no reinstalling.

Design rules — this must NEVER stop the app from launching:
  * Not a git clone (ZIP install), git missing, offline, or the pull
    fails for any reason -> print one gentle line and continue.
  * Fast-forward only: local edits are never overwritten or merged.
  * Dependencies reinstall only when requirements.txt actually changed
    (hash marker stored inside .venv, which is per-machine anyway).
  * Set PIPM_NO_UPDATE=1 to skip entirely (e.g. offline shop machine).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKER = Path(sys.prefix) / "pipm-requirements.sha1"  # inside .venv


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def main() -> None:
    if os.environ.get("PIPM_NO_UPDATE"):
        return
    if not (ROOT / ".git").exists():
        return  # ZIP install: updating means re-downloading, don't try

    proc = _run(["git", "pull", "--ff-only", "--quiet"], timeout=45)
    if proc is None:
        print("[update] git unavailable or timed out; starting as-is")
        return
    if proc.returncode != 0:
        # Offline, or local commits/edits diverged. Both are fine to run on.
        print("[update] could not update (offline or local changes); starting as-is")
        return

    req = ROOT / "requirements.txt"
    digest = hashlib.sha1(req.read_bytes()).hexdigest()
    old = MARKER.read_text().strip() if MARKER.exists() else ""
    if digest != old:
        print("[update] dependencies changed; installing (one-time, ~a minute)...")
        pip = _run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
            timeout=900,
        )
        if pip is not None and pip.returncode == 0:
            MARKER.write_text(digest)
            print("[update] dependencies up to date")
        else:
            print("[update] dependency install failed; app may still run "
                  "on the previous versions. Run update.bat when online.")
    else:
        MARKER.write_text(digest)


if __name__ == "__main__":
    main()
