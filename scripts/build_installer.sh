#!/bin/bash
set -e

# Script to build the self-extracting installer

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")
BUILD_DIR="$REPO_ROOT/build-output"
PAYLOAD_ROOT="${1:-}"
ARTIFACT_NAME="chunksilo-installer.sh"

mkdir -p "$BUILD_DIR"

echo "Building installer..."

# Pick a payload root that already contains pre-downloaded wheels so the
# installer remains fully offline.
if [ -z "$PAYLOAD_ROOT" ]; then
    if [ -d "$REPO_ROOT/release_package_manylinux_2_34/chunksilo" ]; then
        PAYLOAD_ROOT="$REPO_ROOT/release_package_manylinux_2_34/chunksilo"
    elif [ -d "$REPO_ROOT/release_package_manylinux_2_28/chunksilo" ]; then
        PAYLOAD_ROOT="$REPO_ROOT/release_package_manylinux_2_28/chunksilo"
    else
        PAYLOAD_ROOT="$REPO_ROOT"
    fi
fi

if [ ! -d "$PAYLOAD_ROOT/dependencies" ] || [ -z "$(ls -A "$PAYLOAD_ROOT/dependencies" 2>/dev/null)" ]; then
    echo "Error: no packaged dependencies found at $PAYLOAD_ROOT/dependencies."
    echo "Run the manylinux packaging scripts first (build-all.sh/package-manylinux-2_34.sh or 2_28.sh) so wheels are included."
    exit 1
fi

echo "Using payload root: $PAYLOAD_ROOT"

# Create a zip of the payload contents, excluding git and other non-essentials
# We cd to payload root to keep paths clean
cd "$PAYLOAD_ROOT"

# Ensure zip is not included in itself if we rerun
rm -f "$BUILD_DIR/payload.zip"

echo "Creating payload zip..."
zip -r -q "$BUILD_DIR/payload.zip" . -x "*.git*" "venv*" "build-output*" "__pycache__*" ".DS_Store" "test*" ".pytest_cache*"

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
