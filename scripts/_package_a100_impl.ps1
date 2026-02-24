#Requires -Version 5.1
<#
.SYNOPSIS
    PuffinZipAI — A100 PCIe Pod Packaging (implementation).
.DESCRIPTION
    Called by package_a100.bat. Creates a deployment-ready ZIP with all files
    needed to run PuffinZipAI on an A100 PCIe Linux pod.

    Excludes: .venv, __pycache__, *.pyc, logs/*, data/models/*, checkpoint data,
              gui_state.json, preflight result JSONs, .git
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ── Resolve paths ────────────────────────────────────────────────────────────
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$Datestamp   = Get-Date -Format 'yyyy-MM-dd_HHmm'
$ZipName     = "PuffinZipAI_A100_$Datestamp.zip"
$ZipPath     = Join-Path $ProjectDir $ZipName

Write-Host ''
Write-Host '============================================================'
Write-Host '  PuffinZipAI A100 PCIe Pod Packager'
Write-Host '============================================================'
Write-Host ''
Write-Host "  Project:  $ProjectDir"
Write-Host "  Output:   $ZipName"
Write-Host ''

# ── 1. Pre-clean __pycache__ (project only, skip .venv) ─────────────────────
Write-Host '[1/4] Cleaning __pycache__ and .pyc files...'
Get-ChildItem -Path $ProjectDir -Directory -Recurse -Filter '__pycache__' |
    Where-Object { $_.FullName -notlike '*\.venv\*' } |
    ForEach-Object { Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue }
Get-ChildItem -Path $ProjectDir -Recurse -Filter '*.pyc' |
    Where-Object { $_.FullName -notlike '*\.venv\*' } |
    Remove-Item -Force -ErrorAction SilentlyContinue
Write-Host '  Done.'

# ── 2. Build file list ──────────────────────────────────────────────────────
Write-Host '[2/4] Building file list...'

$files = [System.Collections.Generic.List[string]]::new()

# --- Root-level entry points & config ---
$rootFiles = @(
    'main_cli.py',
    'run_gui.py',
    'webui_server.py',
    'webui_theme_manager.py',
    'requirements.txt',
    'CODEBASE_INDEX.md'
)
foreach ($f in $rootFiles) {
    $full = Join-Path $ProjectDir $f
    if (Test-Path $full) { $files.Add($f) }
}

# --- Helper: recursively collect files by extension, relative to project ---
function Add-RecursiveFiles {
    param(
        [string]$SubDir,
        [string[]]$Include
    )
    $base = Join-Path $ProjectDir $SubDir
    if (-not (Test-Path $base)) { return }
    Get-ChildItem -Path $base -Recurse -File -Include $Include |
        Where-Object { $_.FullName -notlike '*\__pycache__\*' } |
        ForEach-Object {
            $rel = $_.FullName.Substring($ProjectDir.Length + 1)
            $files.Add($rel)
        }
}

# Core AI package
Add-RecursiveFiles 'puffinzip_ai' @('*.py', '*.rs', '*.cu')

# GUI package
Add-RecursiveFiles 'puffinzip_gui' @('*.py', '*.json')

# Web UI assets
Add-RecursiveFiles 'webui_static'    @('*.*')
Add-RecursiveFiles 'webui_templates' @('*.*')

# Docs
Add-RecursiveFiles 'docs' @('*.*')

# Examples
Add-RecursiveFiles 'examples' @('*.py')

# Tests
Add-RecursiveFiles 'tests' @('*.py')

# --- Scripts (explicit list — no result JSONs) ---
$scriptFiles = @(
    'scripts\run_webui_windows.bat',
    'scripts\run_gui.spec',
    'scripts\preflight_metrics_check.py',
    'scripts\package_a100.bat',
    'scripts\_package_a100_impl.ps1'
)
foreach ($f in $scriptFiles) {
    $full = Join-Path $ProjectDir $f
    if (Test-Path $full) { $files.Add($f) }
}

# Universal launchers (repo root)
foreach ($launcher in @('start.sh', 'start.bat')) {
    $full = Join-Path $ProjectDir $launcher
    if (Test-Path $full) { $files.Add($launcher) }
}
foreach ($f in $scriptFiles) {
    $full = Join-Path $ProjectDir $f
    if (Test-Path $full) { $files.Add($f) }
}

# --- Checkpoint index (schema only, no checkpoint data) ---
$cpIdx = 'checkpoints\checkpoint_index.json'
if (Test-Path (Join-Path $ProjectDir $cpIdx)) { $files.Add($cpIdx) }

Write-Host "  $($files.Count) files to package."

# ── 3. Create ZIP ───────────────────────────────────────────────────────────
Write-Host "[3/4] Creating $ZipName ..."

# Stage into temp directory to get clean folder structure in ZIP
$staging = Join-Path $env:TEMP "PuffinZipAI_A100_staging_$([guid]::NewGuid().ToString('N').Substring(0,8))"
New-Item $staging -ItemType Directory -Force | Out-Null

foreach ($rel in $files) {
    $src = Join-Path $ProjectDir $rel
    $dst = Join-Path $staging  $rel
    $dstDir = Split-Path $dst -Parent
    if (-not (Test-Path $dstDir)) {
        New-Item $dstDir -ItemType Directory -Force | Out-Null
    }
    Copy-Item $src $dst -Force
}

# Remove old ZIP if present
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }

Compress-Archive -Path (Join-Path $staging '*') -DestinationPath $ZipPath -Force

# Cleanup staging (non-fatal — temp dir will be cleaned by OS if locked)
try { Remove-Item $staging -Recurse -Force } catch { }

if (-not (Test-Path $ZipPath)) {
    Write-Host ''
    Write-Host '[ERROR] ZIP creation failed!' -ForegroundColor Red
    exit 1
}

$sizeMB = [math]::Round((Get-Item $ZipPath).Length / 1MB, 2)
Write-Host "  ZIP created: $sizeMB MB"

# ── 4. Summary ──────────────────────────────────────────────────────────────
Write-Host '[4/4] Done!'
Write-Host ''
Write-Host '============================================================'
Write-Host "  Package ready: $ZipName"
Write-Host "  Location:      $ZipPath"
Write-Host ''
Write-Host '  Deploy to pod:'
Write-Host '    1. Upload ZIP to the pod'
Write-Host "    2. unzip $ZipName"
Write-Host '    3. cd PuffinZipAI && bash start.sh'
Write-Host '============================================================'
