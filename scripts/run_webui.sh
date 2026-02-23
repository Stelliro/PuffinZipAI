#!/bin/bash
# ============================================================================
# PuffinZipAI Web UI Launcher - Linux/macOS
# ============================================================================

set -u

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Flask not installed. Installing..."
    python3 -m pip install flask flask-cors
fi

# Function to open browser
open_browser() {
    if command -v xdg-open &> /dev/null; then
        xdg-open http://localhost:5000 >/dev/null 2>&1 &
    elif command -v open &> /dev/null; then
        open http://localhost:5000 >/dev/null 2>&1 &
    fi
}

# Print startup message
cat << EOF

╔════════════════════════════════════════════════════════╗
║        PuffinZipAI Web UI - Starting Server             ║
╠════════════════════════════════════════════════════════╣
║  Waiting for server readiness before opening browser   ║
║                                                        ║
║  Press Ctrl+C to stop the server                       ║
╚════════════════════════════════════════════════════════╝

EOF

cd "$SCRIPT_DIR/.."

SERVER_LOG="${TMPDIR:-/tmp}/puffinzip_webui_server_${RANDOM}_${RANDOM}.log"
python3 webui_server.py --host 127.0.0.1 --port 5000 > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        kill "$SERVER_PID" >/dev/null 2>&1
    fi
}
trap cleanup EXIT INT TERM

SPINNER='|/-\\'
WAIT_SECONDS=0
MAX_WAIT_SECONDS=90

while true; do
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        echo
        echo "ERROR: Web UI server exited before becoming ready."
        if [ -f "$SERVER_LOG" ]; then
            echo "----- Last server log lines -----"
            tail -n 40 "$SERVER_LOG"
        fi
        exit 1
    fi

    if python3 - <<'PY'
import sys
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:5000/api/status", timeout=1):
        pass
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
    then
        break
    fi

    SPIN_CHAR=${SPINNER:$((WAIT_SECONDS % 4)):1}
    printf "\rWaiting for server to start %s (%ss)" "$SPIN_CHAR" "$WAIT_SECONDS"
    sleep 1
    WAIT_SECONDS=$((WAIT_SECONDS + 1))

    if [ "$WAIT_SECONDS" -ge "$MAX_WAIT_SECONDS" ]; then
        echo
        echo "ERROR: Web UI did not become ready within ${MAX_WAIT_SECONDS} seconds."
        if [ -f "$SERVER_LOG" ]; then
            echo "----- Last server log lines -----"
            tail -n 40 "$SERVER_LOG"
        fi
        exit 1
    fi
done

echo
echo "Server is ready. Opening browser at http://localhost:5000 ..."
open_browser
echo "Web UI server is running (PID: ${SERVER_PID}). Press Ctrl+C to stop."

wait "$SERVER_PID"
