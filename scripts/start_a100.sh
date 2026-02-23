#!/usr/bin/env bash
# ============================================================================
#  PuffinZipAI — A100 PCIe Pod Launcher (Multi-GPU + Multi-CPU)
#
#  Self-contained: copy this file to any Linux system with Python 3.10+ and
#  run it.  It will clone the repo, set up a venv, install deps, and start
#  the WebUI — all in one command.
#
#  A100 PCIe — 40/80 GB VRAM, 312 TFLOPS TF32, 624 TFLOPS FP16 (w/ sparsity)
#  Recommended session: ≤7 hours (enough for 2 full training runs).
#
#  Console stays open for live logs & debugging.
#
#  Environment variables (all optional):
#    PUFFIN_PORT          WebUI port              (default: 5001)
#    PUFFIN_HOST          Bind address             (default: 0.0.0.0)
#    PUFFIN_WORKERS       CPU worker count         (default: auto = cores - 1)
#    PUFFIN_GPUS          Comma-separated GPU IDs  (default: all available)
#    PUFFIN_CACHE_MAX_MB  GitHub cache limit in MB (default: 200)
#    PUFFIN_CACHE_MAX_FILES  Max cached files      (default: 500)
#    GITHUB_TOKEN         GitHub API token         (optional, higher rate limits)
#    PUFFIN_REPO_URL      Override repo clone URL
#    PUFFIN_REPO_BRANCH   Branch to checkout       (default: main)
# ============================================================================
set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
REPO_URL="${PUFFIN_REPO_URL:-https://github.com/Stelliro/PuffinZipAI.git}"
REPO_BRANCH="${PUFFIN_REPO_BRANCH:-main}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PUFFIN_PORT:-5001}"
HOST="${PUFFIN_HOST:-0.0.0.0}"
WORKERS="${PUFFIN_WORKERS:-0}"
CACHE_MAX_MB="${PUFFIN_CACHE_MAX_MB:-200}"
CACHE_MAX_FILES="${PUFFIN_CACHE_MAX_FILES:-500}"

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

banner() {
cat << 'EOF'

  ╔══════════════════════════════════════════════════════════╗
  ║             🐧  PuffinZipAI  v0.9.7                     ║
  ║    A100 PCIe · Multi-GPU · Multi-CPU · DQN Neural Agents║
  ╠══════════════════════════════════════════════════════════╣
  ║  WebUI will start on the port shown below.               ║
  ║  This console stays open for logs & debugging.           ║
  ║  Press Ctrl+C to stop the server.                        ║
  ╚══════════════════════════════════════════════════════════╝

EOF
}

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; }
info() { echo -e "${CYAN}[i]${NC} $*"; }

# ── 0. Clone repo if needed ─────────────────────────────────────────────────
# Detect project directory: if this script lives inside scripts/, use parent.
# Otherwise (standalone copy), clone into ./PuffinZipAI beside the script.
if [[ -f "$SCRIPT_DIR/../puffinzip_ai/__init__.py" ]]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
    log "Project found at ${BOLD}$PROJECT_DIR${NC}"
else
    PROJECT_DIR="$SCRIPT_DIR/PuffinZipAI"
    if [[ ! -d "$PROJECT_DIR" ]]; then
        info "Project not found — cloning from ${BOLD}$REPO_URL${NC} ..."
        if ! command -v git &>/dev/null; then
            err "git is not installed. Please install git and re-run."
            exit 1
        fi
        git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$PROJECT_DIR"
        log "Repository cloned to $PROJECT_DIR"
    else
        info "Updating existing clone..."
        (cd "$PROJECT_DIR" && git pull --ff-only 2>/dev/null) || warn "git pull failed — using existing code"
        log "Project at ${BOLD}$PROJECT_DIR${NC}"
    fi
fi

VENV_DIR="$PROJECT_DIR/.venv"

# ── 1. System checks ────────────────────────────────────────────────────────
banner

info "Project directory: ${BOLD}$PROJECT_DIR${NC}"

# Python
PYTHON=""
for candidate in python3.11 python3.10 python3.12 python3.13 python3; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
if [[ -z "$PYTHON" ]]; then
    err "Python 3 not found. Install python3 and re-run."
    exit 1
fi
PY_VER=$("$PYTHON" --version 2>&1)
log "Found $PY_VER ($PYTHON)"

# ── GPU detection (multi-GPU) ───────────────────────────────────────────────
GPU_COUNT=0
HAS_A100=false
if command -v nvidia-smi &>/dev/null; then
    # Count all available GPUs
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
    GPU_COUNT=${GPU_COUNT// /}  # trim whitespace

    if [[ "$GPU_COUNT" -gt 0 ]]; then
        echo ""
        info "Detected ${BOLD}$GPU_COUNT GPU(s)${NC}:"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | while IFS= read -r line; do
            echo -e "    ${CYAN}→${NC} GPU $line"
            if echo "$line" | grep -qi "A100"; then
                HAS_A100=true
            fi
        done

        # Verify A100 presence
        GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "")
        if echo "$GPU_NAMES" | grep -qi "A100"; then
            HAS_A100=true
        fi
        if [[ "$HAS_A100" != "true" ]]; then
            warn "Expected A100 but found: $(echo "$GPU_NAMES" | head -1) — script will still work"
        fi

        # Set CUDA_VISIBLE_DEVICES — use PUFFIN_GPUS if set, else all GPUs
        if [[ -n "${PUFFIN_GPUS:-}" ]]; then
            export CUDA_VISIBLE_DEVICES="$PUFFIN_GPUS"
            info "Using GPUs (from PUFFIN_GPUS): ${BOLD}$CUDA_VISIBLE_DEVICES${NC}"
        else
            # Build comma-separated list: 0,1,2,...
            ALL_GPU_IDS=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',' | sed 's/,$//')
            export CUDA_VISIBLE_DEVICES="$ALL_GPU_IDS"
            info "Using all GPUs: ${BOLD}$CUDA_VISIBLE_DEVICES${NC}"
        fi
    else
        warn "nvidia-smi found but no GPUs detected — running in CPU-only mode"
    fi
else
    warn "nvidia-smi not found — running in CPU-only mode"
fi

# ── CPU worker auto-detection ───────────────────────────────────────────────
if [[ "$WORKERS" == "0" ]]; then
    CPU_CORES=$(nproc 2>/dev/null || echo 4)
    # Use N-1 cores (leave 1 for system), min 2
    WORKERS=$(( CPU_CORES > 2 ? CPU_CORES - 1 : 2 ))
    info "Auto-detected $CPU_CORES CPU cores → using ${BOLD}$WORKERS workers${NC}"
fi

# ── Cache eviction limits ───────────────────────────────────────────────────
export PUFFIN_CACHE_MAX_MB="$CACHE_MAX_MB"
export PUFFIN_CACHE_MAX_FILES="$CACHE_MAX_FILES"
info "GitHub cache limits: ${BOLD}${CACHE_MAX_MB} MB / ${CACHE_MAX_FILES} files${NC} (LRU eviction)"

# ── 2. Virtual environment ──────────────────────────────────────────────────
cd "$PROJECT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
    log "Virtual environment created at $VENV_DIR"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
log "Activated venv ($(python --version))"

# ── 3. Install dependencies ─────────────────────────────────────────────────
_need_install=0

# Quick check: can we import the critical packages?
python -c "import flask, flask_cors, numpy, psutil, torch" 2>/dev/null || _need_install=1

if [[ "$_need_install" == "1" ]]; then
    info "Installing dependencies (first run)..."

    # Upgrade pip
    python -m pip install --upgrade pip -q

    # Core deps (always needed)
    python -m pip install flask flask-cors numpy psutil requests matplotlib -q
    log "Core packages installed"

    # PyTorch — detect CUDA version and install matching torch
    if command -v nvidia-smi &>/dev/null && [[ "$GPU_COUNT" -gt 0 ]]; then
        info "Installing PyTorch with CUDA support..."
        # PyTorch 2.x with CUDA 12.x (covers A40, A100, H100, 4090, etc.)
        python -m pip install torch --index-url https://download.pytorch.org/whl/cu121 -q 2>/dev/null \
            || python -m pip install torch -q  # fallback to default
        log "PyTorch installed (CUDA)"

        # CuPy for GPU RLE (optional, non-fatal)
        python -m pip install cupy-cuda12x -q 2>/dev/null && log "CuPy installed" \
            || warn "CuPy install failed (optional — GPU RLE disabled)"
    else
        info "Installing PyTorch (CPU-only)..."
        python -m pip install torch --index-url https://download.pytorch.org/whl/cpu -q
        log "PyTorch installed (CPU)"
    fi

    log "All dependencies installed"
else
    log "Dependencies already installed"
fi

# ── 4. Verify imports ───────────────────────────────────────────────────────
info "Verifying PuffinZipAI imports..."
python -c "
import os
from puffinzip_ai import PuffinZipAI
import torch

model_type = getattr(PuffinZipAI, 'MODEL_TYPE', 'tabular')
device = 'CUDA' if torch.cuda.is_available() else 'CPU'
print(f'  Agent: {PuffinZipAI.__name__} ({model_type})')
print(f'  PyTorch device: {device}')

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f'  GPUs available to PyTorch: {gpu_count}')
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_mem / 1024**3
        print(f'    GPU {i}: {name} ({vram:.1f} GB VRAM)')
        if vram >= 38:
            print(f'    A100 VRAM OK — plenty of room for large populations')
else:
    print('  No CUDA GPUs visible to PyTorch')

max_mb = os.environ.get('PUFFIN_CACHE_MAX_MB', '200')
max_files = os.environ.get('PUFFIN_CACHE_MAX_FILES', '500')
print(f'  Cache limits: {max_mb} MB / {max_files} files')
" || { err "Import verification failed! Check the console output above."; exit 1; }
log "All imports OK"

# ── 5. Pre-run cache cleanup ────────────────────────────────────────────────
info "Running pre-flight cache check..."
python -c "
from puffinzip_ai.utils.github_file_fetcher import GitHubFileFetcher
fetcher = GitHubFileFetcher()
stats = fetcher.cache_stats()
print(f'  Cached files: {stats[\"total_files\"]} / {stats[\"max_files\"]}')
print(f'  Cache size:   {stats[\"total_bytes\"] / 1024 / 1024:.1f} MB / {stats[\"max_bytes\"] / 1024 / 1024:.0f} MB')
evicted = fetcher._evict_if_needed()
if evicted:
    print(f'  Evicted {evicted} stale/excess files')
" 2>/dev/null || warn "Cache pre-flight check skipped (non-fatal)"

# ── 6. Start server ─────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Starting PuffinZipAI WebUI${NC}"
echo -e "  ${CYAN}URL:${NC}      ${BOLD}http://$HOST:$PORT${NC}"
echo -e "  ${CYAN}Workers:${NC}  ${BOLD}$WORKERS CPU workers${NC}"
if [[ "$GPU_COUNT" -gt 0 ]]; then
echo -e "  ${CYAN}GPUs:${NC}     ${BOLD}$GPU_COUNT × A100 PCIe (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)${NC}"
else
echo -e "  ${CYAN}GPUs:${NC}     ${BOLD}None (CPU-only mode)${NC}"
fi
echo -e "  ${CYAN}Cache:${NC}    ${BOLD}${CACHE_MAX_MB} MB / ${CACHE_MAX_FILES} files max (LRU eviction)${NC}"
echo -e "  ${CYAN}Console:${NC}  Live logs below — Ctrl+C to stop"
echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo ""

# Export worker count so the WebUI can suggest it as default
export PUFFIN_DEFAULT_WORKERS="$WORKERS"

# Run with console output (no backgrounding, no nohup)
exec python webui_server.py --host "$HOST" --port "$PORT"
