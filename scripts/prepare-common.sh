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

# Copy common files that are the same for all platforms
cp chunksilo.py "$COMMON_ROOT/"
cp cfgload.py "$COMMON_ROOT/"
cp index.py "$COMMON_ROOT/"
cp requirements.txt "$COMMON_ROOT/"
cp README.md "$COMMON_ROOT/"
cp config.json "$COMMON_ROOT/"
cp setup.sh "$COMMON_ROOT/"

echo "$VERSION" > "$COMMON_ROOT/VERSION"

# Download the embedding + rerank models once (they're the same for all platforms)
python3.11 index.py --download-models --model-cache-dir "$COMMON_ROOT/models"

# Verify models were downloaded
if [ -d "$COMMON_ROOT/models" ]; then
  echo "Cached retrieval models stored in $COMMON_ROOT/models"
else
  echo "Expected cached retrieval models at $COMMON_ROOT/models but they were not found." >&2
  exit 1
fi

# Create tarball to preserve file permissions during artifact upload/download
# GitHub Actions artifacts don't preserve Unix permissions, but tar does
tar -cvf release_common.tar -C "$COMMON_ROOT" .

echo "Common files prepared in $COMMON_ROOT/"
echo "Tarball created: release_common.tar (preserves file permissions)"