@echo off
REM ============================================================================
REM PuffinZipAI Web UI Launcher - Windows
REM ============================================================================

setlocal enabledelayedexpansion

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

REM Get the directory of this script
set SCRIPT_DIR=%~dp0

REM Check if required packages are installed
echo Checking dependencies...
python -c "import flask" >nul 2>&1
if %errorlevel% neq 0 (
    echo Flask not installed. Installing...
    python -m pip install flask flask-cors
)

REM Start the web server
echo.
echo ╔════════════════════════════════════════════════════════╗
echo ║        PuffinZipAI Web UI - Starting Server             ║
echo ╠════════════════════════════════════════════════════════╣
echo ║  Opening http://localhost:5000 in your browser         ║
echo ║                                                        ║
echo ║  Close this window to stop the server                 ║
echo ╚════════════════════════════════════════════════════════╝
echo.

REM Open browser
start http://localhost:5000

REM Start the server
cd /d "%SCRIPT_DIR%"
python webui_server.py --host 127.0.0.1 --port 5000

pause
