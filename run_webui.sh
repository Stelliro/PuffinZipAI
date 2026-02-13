#!/bin/bash
# ============================================================================
# PuffinZipAI Web UI Launcher - Linux/macOS
# ============================================================================

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
        xdg-open http://localhost:5000
    elif command -v open &> /dev/null; then
        open http://localhost:5000
    fi
}

# Print startup message
cat << EOF

╔════════════════════════════════════════════════════════╗
║        PuffinZipAI Web UI - Starting Server             ║
╠════════════════════════════════════════════════════════╣
║  Opening http://localhost:5000 in your browser         ║
║                                                        ║
║  Press Ctrl+C to stop the server                       ║
╚════════════════════════════════════════════════════════╝

EOF

# Open browser in background
open_browser &

# Start the server
cd "$SCRIPT_DIR"
python3 webui_server.py --host 127.0.0.1 --port 5000
