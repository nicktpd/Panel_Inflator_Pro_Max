@echo off
rem ============================================================
rem  Panel Inflator Pro Max - manual update
rem  Pulls the latest version from GitHub and refreshes
rem  dependencies. run.bat also does this automatically at
rem  launch; use this script to force it or to see errors.
rem  Requires the app to be installed via git clone.
rem ============================================================
setlocal
cd /d "%~dp0"

if not exist ".git" (
    echo [ERROR] This copy was not installed with git, so it cannot
    echo         self-update. Re-download the ZIP from GitHub, or
    echo         install git and clone the repository instead.
    pause
    exit /b 1
)

where git >nul 2>nul
if errorlevel 1 (
    echo [ERROR] git is not installed. Get it from https://git-scm.com
    pause
    exit /b 1
)

echo Pulling latest version...
git pull --ff-only
if errorlevel 1 (
    echo [ERROR] Update failed. If you edited app files locally, revert
    echo         them with:  git checkout -- .   and run this again.
    pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    echo Refreshing dependencies...
    ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
)

echo.
echo Up to date. Double-click run.bat to start the app.
pause
