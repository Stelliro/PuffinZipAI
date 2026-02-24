@echo off
REM ============================================================================
REM PuffinZipAI — Universal Launcher (Windows)
REM
REM Auto-detects hardware (CPU, RAM, GPU type & VRAM) and configures the
REM WebUI accordingly.  Run presets (Test / Medium / Max) are available in
REM the WebUI dashboard.
REM
REM Usage:  start.bat             (normal mode)
REM         start.bat --debug     (debug / verbose mode)
REM
REM Credentials auto-generated into webui_credentials.json on first run.
REM
REM Environment variables (all optional):
REM   PUFFIN_HOST            Bind address          (default: 0.0.0.0)
REM   PUFFIN_PORT            WebUI port            (default: 5001)
REM   PUFFIN_WORKERS         CPU worker count      (default: auto)
REM   PUFFIN_CACHE_MAX_MB    GitHub cache limit    (default: 200)
REM   PUFFIN_CACHE_MAX_FILES Max cached files      (default: 500)
REM   PUFFIN_USERNAME        Override login username
REM   PUFFIN_PASSWORD        Override login password
REM   PUFFIN_SECRET_KEY      Override Flask secret key
REM ============================================================================
setlocal enabledelayedexpansion
title PuffinZipAI — Universal Launcher

REM Navigate to project root (script lives at repo root)
cd /d "%~dp0"

echo.
echo  ========================================================
echo   PuffinZipAI v0.9.8 — Universal Launcher
echo   Auto Hardware Detection + Run Presets
echo  ========================================================
echo.

REM ── Locate Python ─────────────────────────────────────────────────────────
set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else (
    REM Try py launcher, then python on PATH
    py --version >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=py"
    ) else (
        python --version >nul 2>&1
        if !errorlevel! equ 0 (
            set "PYTHON_EXE=python"
        )
    )
)
if not defined PYTHON_EXE (
    echo [ERROR] Python 3 not found. Install Python 3.9+ and re-run.
    pause
    exit /b 1
)
for /f "delims=" %%v in ('%PYTHON_EXE% --version 2^>^&1') do echo [OK] Found %%v

REM ── Create venv if needed ─────────────────────────────────────────────────
if not exist ".venv\Scripts\python.exe" if not exist "venv\Scripts\python.exe" (
    echo [i] Creating virtual environment...
    %PYTHON_EXE% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    set "PYTHON_EXE=.venv\Scripts\python.exe"
    echo [OK] Virtual environment created
)

REM ── Install dependencies if needed ────────────────────────────────────────
%PYTHON_EXE% -c "import flask, flask_cors, numpy, psutil" >nul 2>&1
if errorlevel 1 (
    echo [i] Installing dependencies ^(first run^)...
    %PYTHON_EXE% -m pip install --upgrade pip -q
    %PYTHON_EXE% -m pip install flask flask-cors numpy psutil requests matplotlib -q
    echo [OK] Core packages installed

    REM Attempt PyTorch install
    echo [i] Installing PyTorch...
    %PYTHON_EXE% -m pip install torch -q
    echo [OK] PyTorch installed
    echo [OK] All dependencies installed
) else (
    echo [OK] Dependencies already installed
)

REM ── Hardware detection — GPU ──────────────────────────────────────────────
set "GPU_COUNT=0"
set "GPU_NAME=None"
set "GPU_VRAM_MB=0"

where nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    for /f "usebackq delims=" %%c in (`nvidia-smi --query-gpu^=count --format^=csv^,noheader^,nounits 2^>nul`) do (
        set "GPU_COUNT=%%c"
    )
    REM Trim whitespace
    for /f "tokens=* delims= " %%a in ("!GPU_COUNT!") do set "GPU_COUNT=%%a"

    if !GPU_COUNT! gtr 0 (
        echo.
        echo [i] Detected !GPU_COUNT! GPU^(s^):
        for /f "usebackq delims=" %%g in (`nvidia-smi --query-gpu^=index^,name^,memory.total --format^=csv^,noheader 2^>nul`) do (
            echo     GPU %%g
        )
        REM Get first GPU name
        for /f "usebackq delims=" %%n in (`nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul`) do (
            if "!GPU_NAME!"=="None" set "GPU_NAME=%%n"
        )
        REM Get first GPU VRAM
        for /f "usebackq delims=" %%m in (`nvidia-smi --query-gpu^=memory.total --format^=csv^,noheader^,nounits 2^>nul`) do (
            if "!GPU_VRAM_MB!"=="0" set "GPU_VRAM_MB=%%m"
        )
    ) else (
        echo [!] nvidia-smi found but no GPUs detected — CPU-only mode
    )
) else (
    echo [!] No NVIDIA GPU detected — CPU-only mode
)

REM ── Hardware detection — CPU & RAM (via Python for reliability) ───────────
set "CPU_CORES=4"
set "RAM_MB=8192"

for /f "usebackq delims=" %%c in (`%PYTHON_EXE% -c "import os; print(os.cpu_count() or 4)" 2^>nul`) do set "CPU_CORES=%%c"
for /f "usebackq delims=" %%r in (`%PYTHON_EXE% -c "import psutil; print(int(psutil.virtual_memory().total / 1024 / 1024))" 2^>nul`) do set "RAM_MB=%%r"
set /a "RAM_GB=RAM_MB / 1024"

echo [i] CPU cores: !CPU_CORES! ^| RAM: !RAM_GB! GB

REM ── CPU worker auto-detection ─────────────────────────────────────────────
if not defined PUFFIN_WORKERS (
    set /a "PUFFIN_WORKERS=CPU_CORES - 1"
    if !PUFFIN_WORKERS! lss 2 set "PUFFIN_WORKERS=2"
    echo [i] Auto worker count: !PUFFIN_WORKERS!
)

REM ── Defaults ──────────────────────────────────────────────────────────────
if not defined PUFFIN_PORT set "PUFFIN_PORT=5001"
if not defined PUFFIN_CACHE_MAX_MB set "PUFFIN_CACHE_MAX_MB=200"
if not defined PUFFIN_CACHE_MAX_FILES set "PUFFIN_CACHE_MAX_FILES=500"

REM ── Export hardware profile for WebUI ─────────────────────────────────────
set "PUFFIN_HW_GPU_COUNT=!GPU_COUNT!"
set "PUFFIN_HW_GPU_NAME=!GPU_NAME!"
set "PUFFIN_HW_GPU_VRAM_MB=!GPU_VRAM_MB!"
set "PUFFIN_HW_CPU_CORES=!CPU_CORES!"
set "PUFFIN_HW_RAM_MB=!RAM_MB!"
set "PUFFIN_DEFAULT_WORKERS=!PUFFIN_WORKERS!"

REM ── Ensure credentials ───────────────────────────────────────────────────
echo.
echo [i] Ensuring WebUI credentials...
%PYTHON_EXE% -c "from webui_credentials_manager import load_or_create_credentials, _CREDENTIALS_FILE; c = load_or_create_credentials(); print(f'  Credentials file: {_CREDENTIALS_FILE}'); print(f'  Username: {c[\"username\"]}'); print(f'  Password: {c[\"password\"]}'); print(f'  Public access: {c.get(\"public_access\", False)}')"
if errorlevel 1 (
    echo [ERROR] Failed to load or generate credentials.
    pause
    exit /b 1
)
echo [OK] Credentials ready

REM ── Resolve HOST from credentials if not explicitly set ───────────────────
if not defined PUFFIN_HOST (
    for /f "usebackq delims=" %%h in (`%PYTHON_EXE% -c "from webui_credentials_manager import load_or_create_credentials; c = load_or_create_credentials(); print('0.0.0.0' if c.get('public_access', False) else '127.0.0.1')" 2^>nul`) do set "PUFFIN_HOST=%%h"
    if not defined PUFFIN_HOST set "PUFFIN_HOST=127.0.0.1"
    if "!PUFFIN_HOST!"=="0.0.0.0" (
        echo [i] public_access=true : binding to 0.0.0.0 ^(network-accessible^)
    ) else (
        echo [i] public_access=false : binding to 127.0.0.1 ^(local only^)
    )
)

REM ── Kill previous server on same port ─────────────────────────────────────
for /f %%p in ('powershell -NoProfile -Command "$conn = Get-NetTCPConnection -LocalPort !PUFFIN_PORT! -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; if ($conn) { $conn.OwningProcess }"') do (
    echo [i] Stopping previous server ^(PID: %%p^)...
    taskkill /pid %%p /f >nul 2>&1
    timeout /t 1 /nobreak >nul
)

REM ── Parse --debug flag ────────────────────────────────────────────────────
set "DEBUG_FLAG="
for %%a in (%*) do (
    if /I "%%a"=="--debug" set "DEBUG_FLAG=--debug"
    if /I "%%a"=="-debug" set "DEBUG_FLAG=--debug"
)

REM ── Clear bytecode caches ─────────────────────────────────────────────────
for /d /r "." %%d in (__pycache__) do (
    if exist "%%d" rd /s /q "%%d" >nul 2>&1
)

REM ── Detect public IP (best-effort) ────────────────────────────────────────
set "PUBLIC_IP="
if "!PUFFIN_HOST!"=="0.0.0.0" (
    for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing -Uri 'https://api.ipify.org' -TimeoutSec 3).Content.Trim() } catch { try { (Invoke-WebRequest -UseBasicParsing -Uri 'https://ifconfig.me' -TimeoutSec 3).Content.Trim() } catch { '' } }"`) do set "PUBLIC_IP=%%i"
)

REM ── Start server ──────────────────────────────────────────────────────────
echo.
echo ========================================================
echo   Starting PuffinZipAI WebUI
if defined PUBLIC_IP (
    if not "!PUBLIC_IP!"=="" (
        echo   Connect:  http://!PUBLIC_IP!:!PUFFIN_PORT!
        echo   Bind:     !PUFFIN_HOST!:!PUFFIN_PORT!
    ) else (
        echo   URL:      http://!PUFFIN_HOST!:!PUFFIN_PORT!
    )
) else (
    echo   URL:      http://!PUFFIN_HOST!:!PUFFIN_PORT!
)
echo   Workers:  !PUFFIN_WORKERS! CPU workers
if !GPU_COUNT! gtr 0 (
    set /a "GPU_VRAM_GB=GPU_VRAM_MB / 1024"
    echo   GPUs:     !GPU_COUNT! x !GPU_NAME! ^(!GPU_VRAM_GB! GB VRAM each^)
) else (
    echo   GPUs:     None ^(CPU-only mode^)
)
echo   RAM:      !RAM_GB! GB
echo   CPU:      !CPU_CORES! cores
echo   Cache:    !PUFFIN_CACHE_MAX_MB! MB / !PUFFIN_CACHE_MAX_FILES! files max
echo   Auth:     Enabled ^(credentials in webui_credentials.json^)
if "!PUFFIN_HOST!"=="0.0.0.0" (
    echo   Access:   Public ^(network-accessible^)
) else (
    echo   Access:   Local only ^(127.0.0.1^)
)
echo   Console:  Live logs below
echo ========================================================
echo.

REM Start server in background and wait for health check
set "SERVER_LOG=%TEMP%\puffinzip_webui_%RANDOM%.log"
start "" /b %PYTHON_EXE% webui_server.py --host !PUFFIN_HOST! --port !PUFFIN_PORT! %DEBUG_FLAG% > "%SERVER_LOG%" 2>&1

set /a "WAIT_SECONDS=0"
set /a "MAX_WAIT_SECONDS=90"

:WAIT_FOR_SERVER
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:!PUFFIN_PORT!/health' -TimeoutSec 1 | Out-Null; exit 0 } catch { exit 1 }"
if !errorlevel! equ 0 goto SERVER_READY

set /a "WAIT_SECONDS+=1"
echo Waiting for server to start... (!WAIT_SECONDS!s)
timeout /t 1 /nobreak >nul
if !WAIT_SECONDS! geq !MAX_WAIT_SECONDS! goto SERVER_TIMEOUT
goto WAIT_FOR_SERVER

:SERVER_READY
echo.
echo [OK] Server is ready! Opening browser...
start http://localhost:!PUFFIN_PORT!

set "SERVER_PID="
for /f %%p in ('powershell -NoProfile -Command "$conn = Get-NetTCPConnection -LocalPort !PUFFIN_PORT! -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; if ($conn) { $conn.OwningProcess }"') do set "SERVER_PID=%%p"

echo.
echo PuffinZipAI is running. Press any key to stop the server...
echo Server log: %SERVER_LOG%
type "%SERVER_LOG%" 2>nul
echo.
pause >nul

if defined SERVER_PID (
    echo Stopping server ^(PID: !SERVER_PID!^)...
    taskkill /pid !SERVER_PID! /f >nul 2>&1
)
echo Server stopped. Goodbye!
goto :EOF

:SERVER_TIMEOUT
echo.
echo [ERROR] Server failed to start within !MAX_WAIT_SECONDS!s.
echo Server log output:
type "%SERVER_LOG%" 2>nul
echo.
pause
exit /b 1

:EOF
endlocal
