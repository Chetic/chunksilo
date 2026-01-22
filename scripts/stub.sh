#!/bin/bash
set -e

# Self-extracting installer stub

# Create a temporary directory for extraction.
# Avoid /tmp to support systems where it is not writable.
BASE_TEMP_DIR=${TMPDIR:-"$HOME/.cache/ChunkSilo"}
mkdir -p "$BASE_TEMP_DIR"
TEMP_DIR=$(mktemp -d "$BASE_TEMP_DIR/ChunkSilo-installer.XXXXXXXX")
LAUNCH_DIR=$(pwd)

# Function to cleanup temp dir
cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Find where the payload starts (the line after __PAYLOAD_BEGINS__)
PAYLOAD_START=$(awk '/^__PAYLOAD_BEGINS__$/ {print NR + 1; exit 0; }' "$0")

# Extract the payload (zip file)
tail -n +$PAYLOAD_START "$0" > "$TEMP_DIR/payload.zip"

# Unzip the payload
unzip -q "$TEMP_DIR/payload.zip" -d "$TEMP_DIR"

# Make setup executable and run it, passing on arguments
chmod +x "$TEMP_DIR/setup.sh"

# Run setup.sh with --launch-dir prepended to handle relative paths correctly
"$TEMP_DIR/setup.sh" --launch-dir "$LAUNCH_DIR" "$@"

exit 0

__PAYLOAD_BEGINS__
