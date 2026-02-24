#!/usr/bin/env bash
# ============================================================================
#  PuffinZipAI — Universal Launcher (Linux / macOS)
#
#  Auto-detects hardware (CPU, RAM, GPU type & VRAM) and configures the
#  WebUI accordingly.  Works on any system — from laptops to A100/H100 pods.
#
#  Self-contained: will clone the repo, create a venv, install deps, generate
#  credentials, and start the WebUI — all in one command.
#
#  Environment variables (all optional):
#    PUFFIN_PORT            WebUI port              (default: 5001)
#    PUFFIN_HOST            Bind address             (default: from credentials)
#    PUFFIN_WORKERS         CPU worker count         (default: auto)
#    PUFFIN_GPUS            Comma-separated GPU IDs  (default: all available)
#    PUFFIN_CACHE_MAX_MB    GitHub cache limit in MB (default: 200)
#    PUFFIN_CACHE_MAX_FILES Max cached files         (default: 500)
#    GITHUB_TOKEN           GitHub API token         (optional, higher rate limits)
#    PUFFIN_USERNAME        Override WebUI login username
#    PUFFIN_PASSWORD        Override WebUI login password
#    PUFFIN_SECRET_KEY      Override Flask secret key
#    PUFFIN_ADMIN_USERNAME        Override admin/remote login username
#    PUFFIN_ADMIN_PASSWORD        Override admin/remote login password
#    PUFFIN_CUSTOM_URL            Custom URL shown in banner (e.g. https://stelliro.com/puffinzipai)
#    PUFFIN_REPO_URL        Override repo clone URL
#    PUFFIN_REPO_BRANCH     Branch to checkout       (default: main)
# ============================================================================
set -euo pipefail
# ── Load .env (local-only config, git-ignored) ─────────────────────────────────────────────
_SCRIPT_DIR_EARLY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${_SCRIPT_DIR_EARLY}/.env" ]]; then
    # Export every non-comment, non-empty line
    set -a
    # shellcheck disable=SC1091
    source "${_SCRIPT_DIR_EARLY}/.env"
    set +a
fi
# ── Config ───────────────────────────────────────────────────────────────────
REPO_URL="${PUFFIN_REPO_URL:-https://github.com/Stelliro/PuffinZipAI.git}"
REPO_BRANCH="${PUFFIN_REPO_BRANCH:-main}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PUFFIN_PORT:-5001}"
HOST="${PUFFIN_HOST:-}"
WORKERS="${PUFFIN_WORKERS:-0}"
CACHE_MAX_MB="${PUFFIN_CACHE_MAX_MB:-200}"
CACHE_MAX_FILES="${PUFFIN_CACHE_MAX_FILES:-500}"

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# ── Port availability helper ────────────────────────────────────────────────
_port_in_use() {
    # Returns 0 (true) if port is in use, 1 (false) if free
    if command -v ss &>/dev/null; then
        ss -tlnH "sport = :$1" 2>/dev/null | grep -q .
    elif command -v netstat &>/dev/null; then
        netstat -tlnp 2>/dev/null | grep -q ":$1 "
    else
        # Fallback: try to connect
        (echo >/dev/tcp/127.0.0.1/$1) 2>/dev/null
    fi
}

_find_free_port() {
    # Try preferred port first, then scan candidates
    local preferred="$1"
    shift
    local candidates=("$preferred" "$@")

    for p in "${candidates[@]}"; do
        if ! _port_in_use "$p"; then
            echo "$p"
            return 0
        fi
    done
    # Last resort: ask the OS for an ephemeral port
    local ep
    ep=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()" 2>/dev/null || echo "")
    if [[ -n "$ep" ]]; then
        echo "$ep"
        return 0
    fi
    return 1
}

banner() {
cat << 'EOF'

  ╔══════════════════════════════════════════════════════════╗
  ║              🐧  PuffinZipAI  v0.9.8                    ║
  ║     Universal Launcher · Auto Hardware Detection        ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Hardware is auto-detected and run presets (Test /       ║
  ║  Medium / Max) are available in the WebUI dashboard.     ║
  ║  Press Ctrl+C to stop the server.                        ║
  ╚══════════════════════════════════════════════════════════╝

EOF
}

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; }
info() { echo -e "${CYAN}[i]${NC} $*"; }

# ── 0. Locate or clone project ──────────────────────────────────────────────
# If this script lives inside the project root (presence of puffinzip_ai/), use it.
# If it lives inside scripts/, use parent.  Otherwise clone beside the script.
if [[ -f "$SCRIPT_DIR/puffinzip_ai/__init__.py" ]]; then
    PROJECT_DIR="$SCRIPT_DIR"
elif [[ -f "$SCRIPT_DIR/../puffinzip_ai/__init__.py" ]]; then
    PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
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

# ── Hardware detection — GPU ────────────────────────────────────────────────
GPU_COUNT=0
GPU_NAME="None"
GPU_VRAM_MB=0

if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
    GPU_COUNT=${GPU_COUNT// /}

    if [[ "$GPU_COUNT" -gt 0 ]]; then
        echo ""
        info "Detected ${BOLD}$GPU_COUNT GPU(s)${NC}:"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | while IFS= read -r line; do
            echo -e "    ${CYAN}→${NC} GPU $line"
        done

        # Capture first GPU name and VRAM for hardware profile
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs)
        GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)

        # Set CUDA_VISIBLE_DEVICES
        if [[ -n "${PUFFIN_GPUS:-}" ]]; then
            export CUDA_VISIBLE_DEVICES="$PUFFIN_GPUS"
            info "Using GPUs (from PUFFIN_GPUS): ${BOLD}$CUDA_VISIBLE_DEVICES${NC}"
        else
            ALL_GPU_IDS=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | tr '\n' ',' | sed 's/,$//')
            export CUDA_VISIBLE_DEVICES="$ALL_GPU_IDS"
            info "Using all GPUs: ${BOLD}$CUDA_VISIBLE_DEVICES${NC}"
        fi
    else
        warn "nvidia-smi found but no GPUs detected — CPU-only mode"
    fi
else
    warn "nvidia-smi not found — CPU-only mode"
fi

# ── Hardware detection — CPU & RAM ──────────────────────────────────────────
CPU_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
RAM_MB=$(free -m 2>/dev/null | awk '/^Mem:/{print $2}' || echo 8192)
RAM_GB=$(( RAM_MB / 1024 ))

info "CPU cores: ${BOLD}$CPU_CORES${NC} | RAM: ${BOLD}${RAM_GB} GB${NC}"

# ── CPU worker auto-detection ───────────────────────────────────────────────
if [[ "$WORKERS" == "0" ]]; then
    WORKERS=$(( CPU_CORES > 2 ? CPU_CORES - 1 : 2 ))
    info "Auto-detected $CPU_CORES CPU cores → using ${BOLD}$WORKERS workers${NC}"
fi

# ── Cache eviction limits ───────────────────────────────────────────────────
export PUFFIN_CACHE_MAX_MB="$CACHE_MAX_MB"
export PUFFIN_CACHE_MAX_FILES="$CACHE_MAX_FILES"
info "GitHub cache limits: ${BOLD}${CACHE_MAX_MB} MB / ${CACHE_MAX_FILES} files${NC} (LRU eviction)"

# ── Export hardware profile for the WebUI server ────────────────────────────
export PUFFIN_HW_GPU_COUNT="$GPU_COUNT"
export PUFFIN_HW_GPU_NAME="$GPU_NAME"
export PUFFIN_HW_GPU_VRAM_MB="$GPU_VRAM_MB"
export PUFFIN_HW_CPU_CORES="$CPU_CORES"
export PUFFIN_HW_RAM_MB="$RAM_MB"

# ── 2. Virtual environment ──────────────────────────────────────────────────
cd "$PROJECT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
    log "Virtual environment created at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
log "Activated venv ($(python --version))"

# ── 3. Install dependencies ─────────────────────────────────────────────────
_need_install=0
python -c "import flask, flask_cors, numpy, psutil, torch" 2>/dev/null || _need_install=1

if [[ "$_need_install" == "1" ]]; then
    info "Installing dependencies (first run)..."
    python -m pip install --upgrade pip -q

    # Core deps
    python -m pip install flask flask-cors numpy psutil requests matplotlib -q
    log "Core packages installed"

    # PyTorch — detect CUDA version and install matching torch
    if command -v nvidia-smi &>/dev/null && [[ "$GPU_COUNT" -gt 0 ]]; then
        info "Installing PyTorch with CUDA support..."
        python -m pip install torch --index-url https://download.pytorch.org/whl/cu121 -q 2>/dev/null \
            || python -m pip install torch -q
        log "PyTorch installed (CUDA)"

        # CuPy for GPU RLE (optional)
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
        vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'    GPU {i}: {name} ({vram:.1f} GB VRAM)')
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

# ── 6. Credentials & host resolution ────────────────────────────────────────
info "Ensuring WebUI credentials..."
CRED_OUTPUT=$(python -c "
from webui_credentials_manager import load_or_create_credentials, _CREDENTIALS_FILE
c = load_or_create_credentials()
print(f'  Credentials file: {_CREDENTIALS_FILE}')
print(f'  Username: {c[\"username\"]}')
print(f'  Password: {c[\"password\"]}')
print(f'  Public access: {c.get(\"public_access\", False)}')
# Machine-readable line for the launcher
print(f'PUBLIC_ACCESS={\"1\" if c.get(\"public_access\", False) else \"0\"}')
") || { err "Failed to load or generate credentials."; exit 1; }
echo "$CRED_OUTPUT" | grep -v '^PUBLIC_ACCESS='
log "Credentials ready"

# Resolve HOST from credentials if not explicitly set via env
if [[ -z "$HOST" ]]; then
    if echo "$CRED_OUTPUT" | grep -q 'PUBLIC_ACCESS=1'; then
        HOST="0.0.0.0"
        info "public_access=true → binding to ${BOLD}0.0.0.0${NC} (network-accessible)"
    else
        HOST="127.0.0.1"
        info "public_access=false → binding to ${BOLD}127.0.0.1${NC} (local only)"
    fi
fi

# ── 7. Resolve available port ────────────────────────────────────────────────
# On RunPod, the proxy only forwards ports configured in the pod template.
# GPU pods always expose 8888 (Jupyter) by default — so we use that.
# We kill any existing process on the chosen port to guarantee availability.
PREFERRED_PORT="$PORT"

_kill_port_occupant() {
    local p="$1"
    local pid
    pid=$(fuser "$p/tcp" 2>/dev/null | tr -d '[:space:]')
    if [[ -n "$pid" ]]; then
        local name
        name=$(ps -p "$pid" -o comm= 2>/dev/null || echo "unknown")
        warn "Killing ${BOLD}$name${NC} (PID $pid) on port $p"
        kill -9 "$pid" 2>/dev/null
        sleep 1
    fi
}

if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    # ── RunPod GPU Pod ──
    # Default exposed HTTP ports on RunPod templates: 8888 (always), plus
    # whatever the user added.  The RUNPOD_TCP_PORT_* env vars only exist
    # on serverless endpoints, not GPU pods, so we hardcode the known default.
    RUNPOD_DEFAULT_PORTS=(8888 8080 3000 7860 5001)

    if [[ "$PREFERRED_PORT" != "5001" ]]; then
        # User explicitly set PUFFIN_PORT — honour it, kill occupant
        PORT="$PREFERRED_PORT"
        info "Using explicitly requested port ${BOLD}$PORT${NC}"
    else
        # No explicit port — pick the first RunPod-exposed port we can use
        PORT=""
        for rp in "${RUNPOD_DEFAULT_PORTS[@]}"; do
            if ! _port_in_use "$rp"; then
                PORT="$rp"
                break
            fi
        done
        # All common ports occupied — take over 8888 (most likely to be proxied)
        if [[ -z "$PORT" ]]; then
            PORT=8888
        fi
        if [[ "$PORT" != "$PREFERRED_PORT" ]]; then
            info "RunPod: using proxy-exposed port ${BOLD}$PORT${NC} (default 5001 is not exposed through RunPod's proxy)"
        fi
    fi

    # Kill whatever currently occupies the port (e.g. Jupyter on 8888)
    if _port_in_use "$PORT"; then
        _kill_port_occupant "$PORT"
    fi
else
    # ── Non-RunPod: find a free port from candidates ──
    PORT_CANDIDATES=("$PREFERRED_PORT" 5001 5002 5003 8080 8888 9000)

    # Remove duplicates while preserving order
    declare -A _seen_ports
    UNIQUE_CANDIDATES=()
    for p in "${PORT_CANDIDATES[@]}"; do
        if [[ -z "${_seen_ports[$p]:-}" ]]; then
            _seen_ports[$p]=1
            UNIQUE_CANDIDATES+=("$p")
        fi
    done

    CHOSEN_PORT=$(_find_free_port "${UNIQUE_CANDIDATES[@]}")
    if [[ -z "$CHOSEN_PORT" ]]; then
        err "No available port found. Tried: ${UNIQUE_CANDIDATES[*]}"
        exit 1
    fi
    if [[ "$CHOSEN_PORT" != "$PREFERRED_PORT" ]]; then
        warn "Port ${BOLD}$PREFERRED_PORT${NC} is in use → switching to ${BOLD}$CHOSEN_PORT${NC}"
    fi
    PORT="$CHOSEN_PORT"
fi

info "Using port ${BOLD}$PORT${NC}"

# ── 8. Detect connect URL (RunPod proxy / public IP / local) ────────────────
CONNECT_URL=""
PLATFORM=""
if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    # RunPod: pods are behind NAT — use the RunPod reverse-proxy URL
    PLATFORM="RunPod"
    CONNECT_URL="https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net"
    info "RunPod detected (pod ${BOLD}${RUNPOD_POD_ID}${NC})"
elif [[ -n "${VAST_CONTAINERLABEL:-}" ]]; then
    PLATFORM="Vast.ai"
    # Vast.ai: construct proxy URL if available
    if [[ -n "${VAST_TCP_PORT_MAP:-}" ]]; then
        info "Vast.ai detected — check your Vast dashboard for the connect URL"
    fi
elif [[ "$HOST" == "0.0.0.0" ]]; then
    # Generic cloud / bare metal — try to detect a routable public IP
    PUBLIC_IP=$(curl -s --max-time 3 https://api.ipify.org 2>/dev/null \
             || curl -s --max-time 3 https://ifconfig.me 2>/dev/null \
             || curl -s --max-time 3 https://icanhazip.com 2>/dev/null \
             || echo "")
    PUBLIC_IP=$(echo "$PUBLIC_IP" | tr -d '[:space:]')
    if [[ -n "$PUBLIC_IP" ]]; then
        CONNECT_URL="http://$PUBLIC_IP:$PORT"
    fi
fi

# ── 9. Start server ─────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Starting PuffinZipAI WebUI${NC}"
if [[ -n "$CONNECT_URL" ]]; then
echo -e "  ${CYAN}Connect:${NC}  ${BOLD}$CONNECT_URL${NC}"
echo -e "  ${CYAN}Bind:${NC}     ${BOLD}$HOST:$PORT${NC}"
else
echo -e "  ${CYAN}URL:${NC}      ${BOLD}http://$HOST:$PORT${NC}"
fi
if [[ -n "${PUFFIN_CUSTOM_URL:-}" ]]; then
echo -e "  ${CYAN}Custom:${NC}   ${BOLD}${PUFFIN_CUSTOM_URL}${NC}"
fi
if [[ -n "$PLATFORM" ]]; then
echo -e "  ${CYAN}Platform:${NC} ${BOLD}$PLATFORM${NC}"
fi
echo -e "  ${CYAN}Workers:${NC}  ${BOLD}$WORKERS CPU workers${NC}"
if [[ "$GPU_COUNT" -gt 0 ]]; then
echo -e "  ${CYAN}GPUs:${NC}     ${BOLD}$GPU_COUNT × $GPU_NAME ($((GPU_VRAM_MB / 1024)) GB VRAM each)${NC}"
else
echo -e "  ${CYAN}GPUs:${NC}     ${BOLD}None (CPU-only mode)${NC}"
fi
echo -e "  ${CYAN}RAM:${NC}      ${BOLD}${RAM_GB} GB${NC}"
echo -e "  ${CYAN}CPU:${NC}      ${BOLD}${CPU_CORES} cores${NC}"
echo -e "  ${CYAN}Cache:${NC}    ${BOLD}${CACHE_MAX_MB} MB / ${CACHE_MAX_FILES} files max (LRU eviction)${NC}"
echo -e "  ${CYAN}Auth:${NC}     ${BOLD}Enabled (credentials in webui_credentials.json)${NC}"
if [[ -n "${PUFFIN_ADMIN_USERNAME:-}" && -n "${PUFFIN_ADMIN_PASSWORD:-}" ]]; then
echo -e "  ${CYAN}Admin:${NC}    ${BOLD}Enabled (remote-access login active)${NC}"
else
echo -e "  ${CYAN}Admin:${NC}    ${BOLD}Disabled (set PUFFIN_ADMIN_USERNAME/PASSWORD in .env)${NC}"
fi
if [[ "$HOST" == "0.0.0.0" ]]; then
echo -e "  ${CYAN}Access:${NC}   ${BOLD}Public (network-accessible)${NC}"
else
echo -e "  ${CYAN}Access:${NC}   ${BOLD}Local only (127.0.0.1)${NC}"
fi
echo -e "  ${CYAN}Console:${NC}  Live logs below — Ctrl+C to stop"
echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}"
echo ""

# Export worker count so the WebUI can suggest it as default
export PUFFIN_DEFAULT_WORKERS="$WORKERS"

# Run with console output
exec python webui_server.py --host "$HOST" --port "$PORT"
