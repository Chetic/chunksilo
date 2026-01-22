#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

echo "=== ChunkSilo Installer ==="

# Find Python
find_python() {
    for ver in "3.12" "3.11" "3.10"; do
        if command -v "python$ver" &> /dev/null; then
            echo "python$ver"
            return
        fi
    done
    if command -v python3 &> /dev/null; then
        echo "python3"
        return
    fi
}

PYTHON_CMD=$(find_python)

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python 3.10+ is required but not found." >&2
    exit 1
fi
echo "Using Python: $PYTHON_CMD"

# Setup venv
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
fi
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv

echo "Installing dependencies..."
DEP_DIR="$SCRIPT_DIR/dependencies"

if [ -d "$DEP_DIR" ] && [ -n "$(ls -A "$DEP_DIR" 2>/dev/null)" ]; then
    # Packaged mode: use pre-downloaded wheels
    echo "Using packaged dependencies from $DEP_DIR"
    ./venv/bin/pip install --no-index --find-links "$DEP_DIR" --no-cache-dir wheel
    ./venv/bin/pip install --no-index --find-links "$DEP_DIR" --no-cache-dir -r requirements.txt
else
    # Development mode: install from PyPI
    echo "Development mode: installing dependencies from PyPI..."
    ./venv/bin/pip install --no-cache-dir wheel
    ./venv/bin/pip install --no-cache-dir -r requirements.txt
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "1. Edit config.json to configure your directories and settings"
echo "2. Add documents to ./data (or update indexing.directories in config.json)"
echo "3. Build the index: ./venv/bin/python index.py"
echo "4. Configure your MCP client (see example below)"
echo ""
echo "=== MCP Client Configuration ==="
echo ""
echo "Add this to your MCP client settings:"
echo ""
echo "  Claude Desktop: ~/Library/Application Support/Claude/claude_desktop_config.json"
echo "  Cline:          VS Code settings > Cline > MCP Servers"
echo "  Roo Code:       .roo/mcp.json in your project"
echo ""
cat << EOF
{
  "mcpServers": {
    "chunksilo": {
      "command": "$SCRIPT_DIR/venv/bin/python",
      "args": ["chunksilo.py"],
      "cwd": "$SCRIPT_DIR"
    }
  }
}
EOF
