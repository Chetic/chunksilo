#!/bin/bash
# Convenience script to build manylinux package
# This runs all steps: prepare common files, then build the package

set -eo pipefail

VERSION="${1:-dev}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Building manylinux package for version: $VERSION"
echo ""

# Step 1: Prepare common files
echo "Step 1: Preparing common files..."
cd "$PROJECT_ROOT"
"$SCRIPT_DIR/prepare-common.sh" "$VERSION"
echo ""

# Step 2: Build manylinux_2_34
echo "Step 2: Building manylinux_2_34 package..."
"$SCRIPT_DIR/package-manylinux-2_34.sh" "$VERSION"
echo ""

echo "Package built successfully!"
echo "Output files:"
ls -lh "$PROJECT_ROOT/build-output/"*.zip 2>/dev/null || echo "  (no zip files found)"