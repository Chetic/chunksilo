#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

echo "=== ChunkSilo Setup ==="

if ! command -v python3.11 &> /dev/null; then
    echo "Error: python3.11 is required but not found." >&2
    exit 1
fi

# Setup venv
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
fi
echo "Creating virtual environment..."
python3.11 -m venv venv

echo "Installing dependencies..."
DEP_DIR="$SCRIPT_DIR/dependencies"

if [ -d "$DEP_DIR" ] && [ -n "$(ls -A "$DEP_DIR" 2>/dev/null)" ]; then
    # Packaged mode: use pre-downloaded wheels
    echo "Using packaged dependencies from $DEP_DIR"
    ./venv/bin/pip install --no-index --find-links "$DEP_DIR" --no-cache-dir -r requirements.txt llama-index-readers-confluence
else
    # Development mode: install from PyPI
    echo "Development mode: installing dependencies from PyPI..."
    ./venv/bin/pip install --no-cache-dir -r requirements.txt llama-index-readers-confluence
fi

# Install the package itself (entry points, no deps since they're already installed)
./venv/bin/pip install --no-deps .

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Edit config.yaml to configure your directories and settings"
echo "2. Add documents to ./data (or update indexing.directories in config.yaml)"
echo "3. Build the index: ./venv/bin/python -m chunksilo.index"
echo "4. Configure your MCP client (see example below)"
echo ""
echo "=== MCP Client Configuration ==="
echo ""
echo "Add this to your MCP client settings:"
echo ""
cat << EOF
{
  "mcpServers": {
    "chunksilo": {
      "command": "$SCRIPT_DIR/venv/bin/python",
      "args": ["-m", "chunksilo.server"],
      "cwd": "$SCRIPT_DIR"
    }
  }
}
EOF
