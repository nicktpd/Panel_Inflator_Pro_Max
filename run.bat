@echo off
rem ============================================================
rem  Panel Inflator Pro Max - daily driver
rem  Starts the local server on 127.0.0.1:8177 and opens the app.
rem ============================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] No virtual environment found. Run setup.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m backend.main
if errorlevel 1 pause
