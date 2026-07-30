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
    bar = "=" * 52
    print(bar)
    print("  Panel Inflator Pro Max - checking for updates")
    print(bar)

    if os.environ.get("PIPM_NO_UPDATE"):
        print("  Skipped (PIPM_NO_UPDATE is set).")
        print(bar)
        return
    if not (ROOT / ".git").exists():
        # ZIP install: updating means re-downloading, don't try.
        print("  This copy was installed from a ZIP, so it can't")
        print("  auto-update. Re-download the latest ZIP, or install")
        print("  with 'git clone' to get automatic updates.")
        print(bar)
        return

    before = _run(["git", "rev-parse", "--short", "HEAD"], timeout=5)
    before_hash = before.stdout.strip() if before else "?"
    print(f"  Current build: {before_hash}")
    print("  Contacting GitHub ...")

    proc = _run(["git", "pull", "--ff-only"], timeout=45)
    if proc is None:
        print("  Could not run git (not installed or timed out).")
        print("  Starting the version you have.")
        print(bar)
        return
    if proc.returncode != 0:
        print("  No update applied (offline, or you have local edits).")
        print("  Starting the version you have.")
        print(bar)
        return

    after = _run(["git", "rev-parse", "--short", "HEAD"], timeout=5)
    after_hash = after.stdout.strip() if after else "?"
    if after_hash == before_hash:
        print("  Already up to date.")
    else:
        subj = _run(["git", "log", "-1", "--format=%s"], timeout=5)
        print(f"  UPDATED: {before_hash} -> {after_hash}")
        if subj and subj.stdout.strip():
            print(f"  Latest change: {subj.stdout.strip()}")

    req = ROOT / "requirements.txt"
    digest = hashlib.sha1(req.read_bytes()).hexdigest()
    old = MARKER.read_text().strip() if MARKER.exists() else ""
    if digest != old:
        print("  Dependencies changed; installing (one-time, ~a minute) ...")
        pip = _run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
            timeout=900,
        )
        if pip is not None and pip.returncode == 0:
            MARKER.write_text(digest)
            print("  Dependencies up to date.")
        else:
            print("  Dependency install failed; run update.bat when online.")
    else:
        MARKER.write_text(digest)
    print(bar)


if __name__ == "__main__":
    main()
