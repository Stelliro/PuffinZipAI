@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Ensure the script runs from the repository root
cd /d "%~dp0\.."

rem Check for .venv first, then venv
set "VENV_DIR=.venv"
if not exist "%VENV_DIR%\Scripts\python.exe" (
    set "VENV_DIR=venv"
)
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

if exist "%PYTHON_EXE%" (
    echo Using existing virtual environment at "%VENV_DIR%".
) else (
    echo Creating virtual environment at ".venv"...
    set "VENV_DIR=.venv"
    set "PYTHON_EXE=.venv\Scripts\python.exe"

    set "PYTHON_CMD="
    rem Try py launcher first, then python on PATH
    py --version >nul 2>&1
    if !errorlevel!==0 (
        set "PYTHON_CMD=py"
    ) else (
        python --version >nul 2>&1
        if !errorlevel!==0 (
            set "PYTHON_CMD=python"
        )
    )

    if not defined PYTHON_CMD (
        echo Failed to find Python. Please ensure Python 3.9+ is installed and on PATH.
        exit /b 1
    )

    !PYTHON_CMD! -m venv ".venv"
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
        call :RepairPip "%PYTHON_EXE%"
    )

    "%PYTHON_EXE%" -m pip install -r "%REQUIREMENTS%"
    if errorlevel 1 (
        echo Failed to install dependencies listed in "%REQUIREMENTS%".
        exit /b 1
    )
) else (
    echo "%REQUIREMENTS%" not found. Skipping dependency installation.
)

echo Clearing Python bytecode caches...
for /d /r "." %%d in (__pycache__) do (
    if exist "%%d" rd /s /q "%%d" >nul 2>&1
)

echo Launching PuffinZip GUI...
"%PYTHON_EXE%" run_gui.py %*
set "EXIT_CODE=%ERRORLEVEL%"

echo PuffinZip GUI exited with code %EXIT_CODE%.

pause
exit /b %EXIT_CODE%

:RepairPip
set "REPAIR_PYTHON=%~1"
echo Warning: Failed to upgrade pip. Attempting to repair pip with ensurepip...
set "REPAIR_HELPER=%TEMP%\pip_repair_%RANDOM%.py"
if exist "%REPAIR_HELPER%" del "%REPAIR_HELPER%" >nul 2>&1
>"%REPAIR_HELPER%" echo import ensurepip
>>"%REPAIR_HELPER%" echo import pathlib
>>"%REPAIR_HELPER%" echo import shutil
>>"%REPAIR_HELPER%" echo import sysconfig
>>"%REPAIR_HELPER%" echo import sys
>>"%REPAIR_HELPER%" echo.
>>"%REPAIR_HELPER%" echo def purge(path):
>>"%REPAIR_HELPER%" echo     if path.is_dir():
>>"%REPAIR_HELPER%" echo         shutil.rmtree(path, ignore_errors=True)
>>"%REPAIR_HELPER%" echo     elif path.exists():
>>"%REPAIR_HELPER%" echo         try:
>>"%REPAIR_HELPER%" echo             path.unlink()
>>"%REPAIR_HELPER%" echo         except FileNotFoundError:
>>"%REPAIR_HELPER%" echo             pass
>>"%REPAIR_HELPER%" echo.
>>"%REPAIR_HELPER%" echo locations = set()
>>"%REPAIR_HELPER%" echo for key in ("purelib", "platlib"):
>>"%REPAIR_HELPER%" echo     value = sysconfig.get_paths().get(key)
>>"%REPAIR_HELPER%" echo     if value:
>>"%REPAIR_HELPER%" echo         locations.add(pathlib.Path(value))
>>"%REPAIR_HELPER%" echo.
>>"%REPAIR_HELPER%" echo for base in list(locations):
>>"%REPAIR_HELPER%" echo     for candidate in base.glob("pip*"):
>>"%REPAIR_HELPER%" echo         name = candidate.name.lower()
>>"%REPAIR_HELPER%" echo         if name == "pip" or name.startswith("pip-") or name.startswith("pip_"):
>>"%REPAIR_HELPER%" echo             purge(candidate)
>>"%REPAIR_HELPER%" echo.
>>"%REPAIR_HELPER%" echo scripts = sysconfig.get_path("scripts")
>>"%REPAIR_HELPER%" echo if scripts:
>>"%REPAIR_HELPER%" echo     for candidate in pathlib.Path(scripts).glob("pip*"):
>>"%REPAIR_HELPER%" echo         purge(candidate)
>>"%REPAIR_HELPER%" echo.
>>"%REPAIR_HELPER%" echo ensurepip.bootstrap(upgrade=True)
"%REPAIR_PYTHON%" "%REPAIR_HELPER%"
set "REPAIR_EXIT=%ERRORLEVEL%"
del "%REPAIR_HELPER%" >nul 2>&1
if not "%REPAIR_EXIT%" == "0" (
    echo Warning: Failed to repair pip via ensurepip. Continuing with the existing version.
    goto :EOF
)
"%REPAIR_PYTHON%" -m pip install --upgrade --force-reinstall pip^<25
if errorlevel 1 (
    echo Warning: Pip upgrade still failing after repair. Continuing with the existing version.
)
goto :EOF
