@echo off
rem ============================================================
rem  Panel Inflator Pro Max - one-time setup (per machine)
rem  Creates a local .venv and installs Python dependencies.
rem  Requires Python 3.11+ installed (python.org, "Add to PATH").
rem ============================================================
setlocal
cd /d "%~dp0"

set PY=
where py >nul 2>nul && set PY=py -3
if "%PY%"=="" (
    where python >nul 2>nul && set PY=python
)
if "%PY%"=="" (
    echo [ERROR] Python not found. Install Python 3.11+ from https://python.org
    echo         and check "Add python.exe to PATH" during install.
    pause
    exit /b 1
)

echo Creating virtual environment in .venv ...
%PY% -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create the virtual environment.
    pause
    exit /b 1
)

echo Installing dependencies (this can take a few minutes) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Check your internet connection and retry.
    pause
    exit /b 1
)

echo.
echo Setup complete. Double-click run.bat to start Panel Inflator Pro Max.
echo Remember: exclude .venv from cloud sync (see SYNC-IGNORE.txt).
pause
