@echo off
setlocal EnableExtensions

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

echo Launching PuffinZip GUI...
"%PYTHON_EXE%" run_gui.py %*
set "EXIT_CODE=%ERRORLEVEL%"

echo PuffinZip GUI exited with code %EXIT_CODE%.

pause
exit /b %EXIT_CODE%

:RepairPip
set "REPAIR_PYTHON=%~1"
echo Warning: Failed to upgrade pip. Attempting to repair pip with ensurepip...
set "REPAIR_SCRIPT=%TEMP%\repair_pip_%RANDOM%%RANDOM%.py"
>"%REPAIR_SCRIPT%" (
    echo import ensurepip
    echo import pathlib
    echo import shutil
    echo import sysconfig
    echo
    echo def _purge(path: pathlib.Path) -> None:
    echo ^    if path.is_dir():
    echo ^        shutil.rmtree(path, ignore_errors=True)
    echo ^    else:
    echo ^        try:
    echo ^            path.unlink()
    echo ^        except FileNotFoundError:
    echo ^            pass
    echo
    echo locations = set()
    echo for key in (^"purelib^", ^"platlib^"):
    echo ^    value = sysconfig.get_paths().get(key)
    echo ^    if value:
    echo ^        locations.add(pathlib.Path(value))
    echo
    echo for base in list(locations):
    echo ^    for candidate in base.glob(^"pip*^"):
    echo ^        name = candidate.name.lower()
    echo ^        if name == ^"pip^" or name.startswith(^"pip-^") or name.startswith(^"pip_^"):
    echo ^            _purge(candidate)
    echo
    echo ensurepip.bootstrap(upgrade=True)
)
"%REPAIR_PYTHON%" "%REPAIR_SCRIPT%"
set "REPAIR_EXIT=%ERRORLEVEL%"
del "%REPAIR_SCRIPT%" >nul 2>&1
if not "%REPAIR_EXIT%" == "0" (
    echo Warning: Failed to repair pip via ensurepip. Continuing with the existing version.
    goto :EOF
)
"%REPAIR_PYTHON%" -m pip install --upgrade --force-reinstall "pip^<25"
if errorlevel 1 (
    echo Warning: Pip upgrade still failing after repair. Continuing with the existing version.
)
goto :EOF
