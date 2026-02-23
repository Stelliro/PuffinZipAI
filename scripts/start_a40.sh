#!/usr/bin/env bash
# ============================================================================
#  PuffinZipAI — A40 Pod Launcher
#  One-command setup + start. Run this every time.
#  Console stays open for live logs & debugging.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PORT="${PUFFIN_PORT:-5001}"
HOST="${PUFFIN_HOST:-0.0.0.0}"
WORKERS="${PUFFIN_WORKERS:-0}"

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

banner() {
cat << 'EOF'

  ╔══════════════════════════════════════════════════════════╗
  ║             🐧  PuffinZipAI  v0.9.6-dev                 ║
  ║          GPU+CPU Pipeline · DQN Neural Agents            ║
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

# GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    log "GPU detected: ${BOLD}$GPU_INFO MB VRAM${NC}"
else
    warn "nvidia-smi not found — running in CPU-only mode"
fi

# Auto-detect CPU workers if not set
if [[ "$WORKERS" == "0" ]]; then
    CPU_CORES=$(nproc 2>/dev/null || echo 4)
    # Use N-1 cores (leave 1 for system), min 2
    WORKERS=$(( CPU_CORES > 2 ? CPU_CORES - 1 : 2 ))
    info "Auto-detected $CPU_CORES CPU cores → using $WORKERS workers"
fi

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
    if command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")
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
from puffinzip_ai import PuffinZipAI
import torch
model_type = getattr(PuffinZipAI, 'MODEL_TYPE', 'tabular')
device = 'CUDA' if torch.cuda.is_available() else 'CPU'
print(f'  Agent: {PuffinZipAI.__name__} ({model_type})')
print(f'  PyTorch device: {device}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
" || { err "Import verification failed! Check the console output above."; exit 1; }
log "All imports OK"

# ── 5. Start server ─────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Starting PuffinZipAI WebUI${NC}"
echo -e "  ${CYAN}URL:${NC}     ${BOLD}http://$HOST:$PORT${NC}"
echo -e "  ${CYAN}Workers:${NC} ${BOLD}$WORKERS CPU workers${NC}"
echo -e "  ${CYAN}Console:${NC} Live logs below — Ctrl+C to stop"
echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo ""

# Export worker count so the WebUI can suggest it as default
export PUFFIN_DEFAULT_WORKERS="$WORKERS"

# Run with console output (no backgrounding, no nohup)
exec python webui_server.py --host "$HOST" --port "$PORT"
