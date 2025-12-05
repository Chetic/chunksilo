#!/bin/bash
# Convenience script to build manylinux packages
# This runs all steps: prepare common files, then build both packages

set -eo pipefail

VERSION="${1:-dev}"
PLATFORM="${2:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Building manylinux packages for version: $VERSION"
echo "Platform: $PLATFORM (options: all, 2_28, 2_34)"
echo ""

# Step 1: Prepare common files
echo "Step 1: Preparing common files..."
cd "$PROJECT_ROOT"
"$SCRIPT_DIR/prepare-common.sh" "$VERSION"
echo ""

# Step 2: Build packages
if [ "$PLATFORM" = "all" ] || [ "$PLATFORM" = "2_34" ]; then
  echo "Step 2a: Building manylinux_2_34 package..."
  "$SCRIPT_DIR/package-manylinux-2_34.sh" "$VERSION"
  echo ""
fi

if [ "$PLATFORM" = "all" ] || [ "$PLATFORM" = "2_28" ]; then
  echo "Step 2b: Building manylinux_2_28 package..."
  "$SCRIPT_DIR/package-manylinux-2_28.sh" "$VERSION"
  echo ""
fi

echo "Package(s) built successfully!"
echo "Output files:"
ls -lh "$PROJECT_ROOT/build-output/"*.zip 2>/dev/null || echo "  (no zip files found)"