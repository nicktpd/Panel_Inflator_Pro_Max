@echo off
rem ============================================================
rem  Panel Inflator Pro Max - daily driver
rem  Auto-updates from GitHub (safe to skip when offline), then
rem  starts the local server on 127.0.0.1:8177 and opens the app.
rem  Set PIPM_NO_UPDATE=1 to launch without checking for updates.
rem ============================================================
rem  NOTE: the body is one parenthesized block on purpose - cmd
rem  parses it fully before executing, so the auto-update can
rem  safely replace this very file while it runs.
setlocal
cd /d "%~dp0"
(
    if not exist ".venv\Scripts\python.exe" (
        echo [ERROR] No virtual environment found. Run setup.bat first.
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m backend.autoupdate
    ".venv\Scripts\python.exe" -m backend.main
    if errorlevel 1 pause
    goto :eof
)
