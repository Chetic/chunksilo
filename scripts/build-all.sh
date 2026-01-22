#!/bin/bash
# Convenience script to build manylinux packages
# This runs all steps: prepare common files, then build both packages

set -eo pipefail

VERSION="${1:-dev}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Building manylinux_2_34 package for version: $VERSION"
echo ""

# Step 1: Prepare common files
echo "Step 1: Preparing common files..."
cd "$PROJECT_ROOT"
"$SCRIPT_DIR/prepare-common.sh" "$VERSION"
echo ""

# Step 2: Build package
echo "Step 2: Building manylinux_2_34 package..."
"$SCRIPT_DIR/package-manylinux-2_34.sh" "$VERSION"
echo ""

echo "Package(s) built successfully!"
echo "Output files:"
ls -lh "$PROJECT_ROOT/build-output/"*.zip 2>/dev/null || echo "  (no zip files found)"