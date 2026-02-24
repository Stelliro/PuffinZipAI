@echo off
REM ============================================================================
REM PuffinZipAI Web UI Launcher - Windows
REM   Usage:  run_webui_windows.bat             (normal mode)
REM           run_webui_windows.bat --debug      (debug/verbose mode)
REM
REM   Credentials are auto-generated into webui_credentials.json on first run.
REM   Environment variables (all optional — override the credentials file):
REM     PUFFIN_HOST         Bind address          (default: 0.0.0.0)
REM     PUFFIN_PORT         WebUI port            (default: 5001)
REM     PUFFIN_USERNAME     Override login username
REM     PUFFIN_PASSWORD     Override login password
REM     PUFFIN_SECRET_KEY   Override Flask secret key
REM ============================================================================

setlocal enabledelayedexpansion

REM Get the directory of this script and move to project root
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%\.."

REM --- Detect venv Python (.venv first, then venv, then system) ---
set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else (
    python --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo ERROR: Python is not installed or not in PATH
        echo Please install Python 3.8+ from https://www.python.org/
        pause
        exit /b 1
    )
    set "PYTHON_EXE=python"
)
echo Using Python: %PYTHON_EXE%

REM --- Host / Port config ---
if not defined PUFFIN_HOST set "PUFFIN_HOST=0.0.0.0"
if not defined PUFFIN_PORT set "PUFFIN_PORT=5001"

REM --- Ensure credentials file exists (auto-generates on first run) ---
echo Ensuring WebUI credentials...
%PYTHON_EXE% -c "from webui_credentials_manager import load_or_create_credentials, _CREDENTIALS_FILE; c = load_or_create_credentials(); print(f'  Credentials file: {_CREDENTIALS_FILE}'); print(f'  Username: {c[\"username\"]}'); print(f'  Password: {c[\"password\"]}')"
if %errorlevel% neq 0 (
    echo ERROR: Failed to load or generate credentials.
    pause
    exit /b 1
)

REM --- Kill any previous PuffinZipAI Web UI server on the configured port ---
echo Checking for previous server instances...
for /f %%p in ('powershell -NoProfile -Command "$conn = Get-NetTCPConnection -LocalPort !PUFFIN_PORT! -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; if ($conn) { $conn.OwningProcess }"') do (
    echo Stopping previous server ^(PID: %%p^)...
    taskkill /pid %%p /f >nul 2>&1
    timeout /t 1 /nobreak >nul
)

REM Check if required packages are installed
echo Checking dependencies...
%PYTHON_EXE% -c "import flask" >nul 2>&1
if %errorlevel% neq 0 (
    echo Flask not installed. Installing...
    %PYTHON_EXE% -m pip install flask flask-cors
)

REM Clear Python bytecode caches so code changes take effect
echo Clearing Python bytecode caches...
for /d /r "." %%d in (__pycache__) do (
    if exist "%%d" rd /s /q "%%d" >nul 2>&1
)

REM Parse --debug flag
set "DEBUG_FLAG="
for %%a in (%*) do (
    if /I "%%a"=="--debug" set "DEBUG_FLAG=--debug"
    if /I "%%a"=="-debug" set "DEBUG_FLAG=--debug"
)

if defined DEBUG_FLAG (
    echo.
    echo  ** DEBUG MODE ENABLED **
    echo.
)

echo.
echo ======================================================
echo   PuffinZipAI Web UI - Starting Server
echo   Binding to !PUFFIN_HOST!:!PUFFIN_PORT!
echo   Authentication: ENABLED (see credentials above)
echo   Waiting for server readiness before opening browser
echo   Press any key later to stop the server
echo ======================================================
echo.

set "SERVER_LOG=%TEMP%\puffinzip_webui_server_%RANDOM%_%RANDOM%.log"

start "" /b %PYTHON_EXE% webui_server.py --host !PUFFIN_HOST! --port !PUFFIN_PORT! %DEBUG_FLAG% > "%SERVER_LOG%" 2>&1

set "SPINNER=|/-\"
set /a "SPIN_INDEX=0"
set /a "WAIT_SECONDS=0"
set /a "MAX_WAIT_SECONDS=90"

:WAIT_FOR_SERVER
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:!PUFFIN_PORT!/health' -TimeoutSec 1 | Out-Null; exit 0 } catch { exit 1 }"
if !errorlevel! equ 0 goto SERVER_READY

set /a "SPIN_INDEX=(SPIN_INDEX+1) %% 4"
set "SPIN_CHAR=|"
if !SPIN_INDEX! equ 1 set "SPIN_CHAR=/"
if !SPIN_INDEX! equ 2 set "SPIN_CHAR=-"
if !SPIN_INDEX! equ 3 set "SPIN_CHAR=+"
echo Waiting for server to start !SPIN_CHAR! (!WAIT_SECONDS!s)
timeout /t 1 /nobreak >nul
set /a "WAIT_SECONDS+=1"
if !WAIT_SECONDS! geq !MAX_WAIT_SECONDS! goto SERVER_TIMEOUT
goto WAIT_FOR_SERVER

:SERVER_READY
echo.
echo Server is ready. Opening browser at http://localhost:!PUFFIN_PORT! ...
start http://localhost:!PUFFIN_PORT!

set "SERVER_PID="
for /f %%p in ('powershell -NoProfile -Command "$conn = Get-NetTCPConnection -LocalPort !PUFFIN_PORT! -State Listen -ErrorAction SilentlyContinue ^| Select-Object -First 1; if ($conn) { $conn.OwningProcess }"') do set "SERVER_PID=%%p"

echo.
if not defined SERVER_PID goto SERVER_READY_NO_PID
echo Web UI server is running (PID: !SERVER_PID!).
goto SERVER_READY_AFTER_PID

:SERVER_READY_NO_PID
echo Web UI server is running.

:SERVER_READY_AFTER_PID
echo Press any key to stop the server...
pause >nul
if defined SERVER_PID taskkill /pid !SERVER_PID! /f >nul 2>&1
exit /b 0

:SERVER_TIMEOUT
echo.
echo ERROR: Web UI did not become ready within !MAX_WAIT_SECONDS! seconds.
for /f %%p in ('powershell -NoProfile -Command "$conn = Get-NetTCPConnection -LocalPort !PUFFIN_PORT! -State Listen -ErrorAction SilentlyContinue ^| Select-Object -First 1; if ($conn) { $conn.OwningProcess }"') do set "SERVER_PID=%%p"
if defined SERVER_PID taskkill /pid !SERVER_PID! /f >nul 2>&1
if exist "%SERVER_LOG%" (
    echo ----- Last server log lines -----
    powershell -NoProfile -Command "if (Test-Path '%SERVER_LOG%') { Get-Content -Path '%SERVER_LOG%' -Tail 40 }"
)
pause
exit /b 1
