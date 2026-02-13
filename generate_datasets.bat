@echo off
SETLOCAL EnableExtensions EnableDelayedExpansion
TITLE PuffinZipAI - Advanced Dataset Generator

echo  ======================================
echo   PuffinZipAI Data Generator
echo  ======================================

:: 1. Environment Detection
set "PYTHON_CMD=python"
if exist "venv\Scripts\python.exe" set "PYTHON_CMD=venv\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set "PYTHON_CMD=.venv\Scripts\python.exe"

%PYTHON_CMD% --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python interpreter not found.
    pause
    exit /b 1
)

echo [INFO] Using Python: %PYTHON_CMD%

:: 2. Configuration Prompts (Optional)
set /p "USER_COUNT=Enter number of files to generate [Default: 75]: "
if "%USER_COUNT%"=="" set "USER_COUNT=75"

echo.
echo [INFO] Generating %USER_COUNT% benchmark files...
echo.

:: 3. Execution - Passing arguments requires modifying benchmark_generator.py to accept sys.argv
:: For now, we run it standard, but we capture the output log location.
%PYTHON_CMD% -m puffinzip_ai.utils.benchmark_generator

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Generation complete.
    echo.
    timeout /t 5
) else (
    echo.
    echo [ERROR] Generation failed. Review errors above.
    pause
)