#!/bin/bash
# Script to prepare common release files (equivalent to the prepare job)
# This creates the release_common directory with all shared files and models

set -eo pipefail

COMMON_ROOT="release_common"
VERSION="${1:-dev}"

echo "Preparing common release files for version: $VERSION"

# Check for Python 3.11 (required)
if ! command -v python3.11 &> /dev/null; then
  echo "Error: python3.11 is required but not found in PATH" >&2
  exit 1
fi

# Create directory
mkdir -p "$COMMON_ROOT"

# Install dependencies so we can download the retrieval models
python3.11 -m pip install --upgrade pip
python3.11 -m pip install -r requirements.txt

# Generate constraints file from installed packages to guide dependency resolution
# This helps pip know what versions work together, reducing backtracking
# Note: pip will find compatible versions for target platform if exact versions aren't available
python3.11 -m pip freeze > "$COMMON_ROOT/minimal-constraints.txt"

# Copy common files that are the same for all platforms
cp mcp_server.py "$COMMON_ROOT/"
cp ingest.py "$COMMON_ROOT/"
cp requirements.txt "$COMMON_ROOT/"
cp README.md "$COMMON_ROOT/"
cp env.template "$COMMON_ROOT/"
cp universal_config.json "$COMMON_ROOT/"
# The installer runs setup.sh and needs generate_configs.py; include them so the
# packaged artifacts remain fully offline and self-contained.
cp setup.sh "$COMMON_ROOT/"
mkdir -p "$COMMON_ROOT/scripts"
cp scripts/generate_configs.py "$COMMON_ROOT/scripts/"

echo "$VERSION" > "$COMMON_ROOT/VERSION"

# Download the embedding + rerank models once (they're the same for all platforms)
export RETRIEVAL_MODEL_CACHE_DIR="$COMMON_ROOT/models"
python3.11 ingest.py --download-models

# Verify models were downloaded
if [ -d "$COMMON_ROOT/models" ]; then
  echo "Cached retrieval models stored in $COMMON_ROOT/models"
else
  echo "Expected cached retrieval models at $COMMON_ROOT/models but they were not found." >&2
  exit 1
fi

echo "Common files prepared in $COMMON_ROOT/"