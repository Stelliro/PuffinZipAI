@echo off
REM ============================================================================
REM  PuffinZipAI — A100 PCIe Pod Packaging Script
REM  Creates a deployment-ready ZIP with all files needed for the A100 pod.
REM  Excludes: .venv, __pycache__, logs/*, data/models/*, checkpoints data,
REM            gui_state.json, *.pyc, preflight result JSONs, .git
REM
REM  Output: PuffinZipAI_A100_<date>.zip  (in project root)
REM  Usage:  scripts\package_a100.bat
REM ============================================================================

REM ── Hand off to PowerShell for reliable execution ──────────────────────────
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0_package_a100_impl.ps1"
