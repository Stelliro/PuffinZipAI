@echo off
setlocal

rem Ensure the script runs from the repository root
cd /d "%~dp0"

set "VENV_DIR=venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

if exist "%PYTHON_EXE%" (
    echo Using existing virtual environment at "%VENV_DIR%".
) else (
    echo Creating virtual environment at "%VENV_DIR%"...
    set "PYTHON_CMD=python"
    py --version >nul 2>&1
    if %errorlevel%==0 (
        set "PYTHON_CMD=py"
    )

    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create virtual environment. Please ensure Python 3.9+ is installed and on PATH.
        exit /b 1
    )
)

if not exist "%PYTHON_EXE%" (
    echo Could not locate the virtual environment interpreter at "%PYTHON_EXE%".
    exit /b 1
)

set "REQUIREMENTS=requirements.txt"
if exist "%REQUIREMENTS%" (
    echo Installing dependencies from "%REQUIREMENTS%"...
    "%PYTHON_EXE%" -m pip install --upgrade pip
    if errorlevel 1 (
        echo Warning: Failed to upgrade pip. Continuing with the existing version.
    )

    "%PYTHON_EXE%" -m pip install -r "%REQUIREMENTS%"
    if errorlevel 1 (
        echo Failed to install dependencies listed in "%REQUIREMENTS%".
        exit /b 1
    )
) else (
    echo "%REQUIREMENTS%" not found. Skipping dependency installation.
)

echo Launching PuffinZip GUI...
"%PYTHON_EXE%" run_gui.py %*
set "EXIT_CODE=%ERRORLEVEL%"

echo PuffinZip GUI exited with code %EXIT_CODE%.

pause
exit /b %EXIT_CODE%
