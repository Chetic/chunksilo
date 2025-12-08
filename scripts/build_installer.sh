#!/bin/bash
set -e

# Script to build the self-extracting installer

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")
BUILD_DIR="$REPO_ROOT/build-output"
ARTIFACT_NAME="opd-mcp-installer.sh"

mkdir -p "$BUILD_DIR"

echo "Building installer..."

# Create a zip of the repo contents, excluding git and other non-essentials
# We cd to repo root to keep paths clean
cd "$REPO_ROOT"

# Ensure zip is not included in itself if we rerun
rm -f "$BUILD_DIR/payload.zip"

echo "Creating payload zip..."
zip -r -q "$BUILD_DIR/payload.zip" . -x "*.git*" "venv*" "build-output*" "__pycache__*" ".DS_Store" "test*" ".pytest_cache*" "config*"

# Determine path to stub
STUB_PATH="$SCRIPT_DIR/stub.sh"

if [ ! -f "$STUB_PATH" ]; then
    echo "Error: stub.sh not found at $STUB_PATH"
    exit 1
fi

# Concatenate stub and payload
OUT_FILE="$BUILD_DIR/$ARTIFACT_NAME"
cat "$STUB_PATH" "$BUILD_DIR/payload.zip" > "$OUT_FILE"
chmod +x "$OUT_FILE"

# Clean up zip
rm "$BUILD_DIR/payload.zip"

echo "Installer created at: $OUT_FILE"
