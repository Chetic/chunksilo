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
python3.11 -m pip install -r requirements.txt llama-index-readers-confluence

# Copy package source and project files
cp -r src "$COMMON_ROOT/"
cp pyproject.toml "$COMMON_ROOT/"
cp requirements.txt "$COMMON_ROOT/"
cp README.md "$COMMON_ROOT/"
cp LICENSE "$COMMON_ROOT/"
cp NOTICE "$COMMON_ROOT/"
cp config.yaml "$COMMON_ROOT/"
cp setup.sh "$COMMON_ROOT/"

echo "$VERSION" > "$COMMON_ROOT/VERSION"

# Stamp the release version into the COPIED sources. The checkout still carries
# the previous release's number here, because the version bump on main only
# happens after a release completes - so without this the bundle would install
# as the previous version even though the tarball and VERSION file say otherwise.
# Done on the copies (not the checkout) so the script is safe to run by hand.
sed -i.bak "s/^version = .*/version = \"$VERSION\"/" "$COMMON_ROOT/pyproject.toml"
sed -i.bak "s/^__version__ = .*/__version__ = \"$VERSION\"/" "$COMMON_ROOT/src/chunksilo/__init__.py"
rm -f "$COMMON_ROOT/pyproject.toml.bak" "$COMMON_ROOT/src/chunksilo/__init__.py.bak"

# Fail loudly rather than ship a bundle whose metadata disagrees with its name.
STAMPED_PROJECT=$(grep -m1 '^version = ' "$COMMON_ROOT/pyproject.toml" | cut -d'"' -f2)
STAMPED_INIT=$(grep -m1 '^__version__ = ' "$COMMON_ROOT/src/chunksilo/__init__.py" | cut -d'"' -f2)
if [ "$STAMPED_PROJECT" != "$VERSION" ] || [ "$STAMPED_INIT" != "$VERSION" ]; then
  echo "Error: version stamping failed (pyproject=$STAMPED_PROJECT, __init__=$STAMPED_INIT, expected $VERSION)" >&2
  exit 1
fi
echo "Stamped version $VERSION into pyproject.toml and __init__.py"

# Download the embedding + rerank models once (they're the same for all
# platforms). The cache location comes from a throwaway config: the CLI has no
# cache-dir flag, and the config given here becomes the active one.
DL_CONFIG=$(mktemp)
printf 'storage:\n  model_cache_dir: "%s"\n' "$COMMON_ROOT/models" > "$DL_CONFIG"
PYTHONPATH=src python3.11 -m chunksilo --download-models --config "$DL_CONFIG"
rm -f "$DL_CONFIG"

# Write model license information
cat > "$COMMON_ROOT/models/MODEL-LICENSES.txt" << 'MODLICEOF'
Bundled Model Licenses
======================

BAAI/bge-small-en-v1.5
  License: MIT
  Copyright: Beijing Academy of Artificial Intelligence (BAAI)
  Source: https://huggingface.co/BAAI/bge-small-en-v1.5

cross-encoder/ms-marco-MiniLM-L-12-v2
  License: Apache-2.0
  Copyright: Nils Reimers
  Source: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2
MODLICEOF

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
