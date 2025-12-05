#!/bin/bash
# Script to build manylinux_2_34 package
# Works both locally (via Docker) and in CI (directly in container)

set -eo pipefail

VERSION="${1:-dev}"
CONTAINER_IMAGE="registry.access.redhat.com/ubi9/ubi:9.6"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# If we're not in GitHub Actions and not in a container, run via Docker
if [ -z "$GITHUB_ACTIONS" ] && [ ! -f /etc/redhat-release ]; then
  # Local execution - use Docker
  echo "Building manylinux_2_34 package for version: $VERSION (via Docker)"
  
  # Check if release_common exists
  if [ ! -d "$PROJECT_ROOT/release_common" ]; then
    echo "Error: release_common directory not found. Run prepare-common.sh first." >&2
    exit 1
  fi
  
  # Create output directory
  OUTPUT_DIR="$PROJECT_ROOT/build-output"
  mkdir -p "$OUTPUT_DIR"
  
  docker run --rm \
    -v "$PROJECT_ROOT:/workspace" \
    -w /workspace \
    -e GITHUB_ACTIONS=1 \
    "$CONTAINER_IMAGE" \
    bash /workspace/scripts/package-manylinux-2_34.sh "$VERSION"
  
  echo "Package built successfully: build-output/opd-mcp-${VERSION}-manylinux_2_34_x86_64.zip"
  exit 0
fi

# We're in CI or already in container - run the packaging logic
echo "Building manylinux_2_34 package for version: $VERSION"

# Check if release_common exists
if [ ! -d "release_common" ]; then
  echo "Error: release_common directory not found." >&2
  exit 1
fi

# Install dependencies including build tools (needed for source distributions)
# Note: Some packages may require additional build tools, but we install the essentials
dnf install -y python3.11 python3.11-pip zip git which gcc gcc-c++ make

# Prepare manylinux_2_34 release package
PACKAGE_ROOT="release_package_manylinux_2_34/opd-mcp"
mkdir -p "$PACKAGE_ROOT"

# Copy common files and models
cp -r release_common/* "$PACKAGE_ROOT/"

# Download dependencies for RHEL 9.6 compatibility
# We're running in UBI 9.6 container which uses glibc 2.34+, compatible with manylinux_2_34
# Strategy: Let pip automatically select compatible wheels for the current system
# This avoids dependency resolution issues from strict platform constraints
python3.11 -m pip install --upgrade pip

echo "Downloading dependencies for RHEL 9.6 (manylinux_2_34 compatible)..."

# Create dependencies directory
mkdir -p "$PACKAGE_ROOT/dependencies"

# Strategy: Download compatible wheels for the current system (RHEL 9.6)
# pip will automatically select wheels compatible with the current platform and Python version
# This is simpler and more reliable than specifying platform constraints
# We allow source distributions as fallback for packages without wheels

# Try 1: Download with constraints (preferred for version consistency)
echo "Attempting to download dependencies with constraints..."
if python3.11 -m pip download \
  -r "$PACKAGE_ROOT/requirements.txt" \
  -c "$PACKAGE_ROOT/minimal-constraints.txt" \
  -d "$PACKAGE_ROOT/dependencies" \
  --no-cache-dir \
  --disable-pip-version-check 2>&1; then
  echo "✓ Successfully downloaded dependencies with constraints"
else
  echo "✗ Failed with constraints, trying without constraints..."
  
  # Try 2: Download without constraints (allows pip to resolve dependencies more freely)
  if python3.11 -m pip download \
    -r "$PACKAGE_ROOT/requirements.txt" \
    -d "$PACKAGE_ROOT/dependencies" \
    --no-cache-dir \
    --disable-pip-version-check 2>&1; then
    echo "✓ Successfully downloaded dependencies without constraints"
  else
    echo "Error: Failed to download dependencies" >&2
    exit 1
  fi
fi

# Verify dependencies were downloaded
if [ ! -d "$PACKAGE_ROOT/dependencies" ] || [ -z "$(ls -A "$PACKAGE_ROOT/dependencies" 2>/dev/null)" ]; then
  echo "Error: Dependencies directory is empty" >&2
  exit 1
fi

echo "✓ Dependencies downloaded successfully to $PACKAGE_ROOT/dependencies"

# Determine output path based on context
if [ -n "$GITHUB_ACTIONS" ]; then
  # In CI, output to current directory
  OUTPUT_ZIP="opd-mcp-${VERSION}-manylinux_2_34_x86_64.zip"
  ZIP_PATH="../$OUTPUT_ZIP"
  FINAL_OUTPUT="$OUTPUT_ZIP"
else
  # Local Docker execution, output to build-output directory
  OUTPUT_DIR="/workspace/build-output"
  mkdir -p "$OUTPUT_DIR"
  OUTPUT_ZIP="$OUTPUT_DIR/opd-mcp-${VERSION}-manylinux_2_34_x86_64.zip"
  # Create zip in current directory first, then move it
  TEMP_ZIP="opd-mcp-${VERSION}-manylinux_2_34_x86_64.zip"
  ZIP_PATH="../$TEMP_ZIP"
  FINAL_OUTPUT="$OUTPUT_ZIP"
fi

# Create zip file
# Ensure we're in the project root (workspace in Docker, PROJECT_ROOT locally)
WORK_DIR="/workspace"
[ ! -d "$WORK_DIR" ] && WORK_DIR="$PROJECT_ROOT"
(cd "$WORK_DIR/release_package_manylinux_2_34" && zip -r "$ZIP_PATH" .)

# Move to final location if needed (for local Docker execution)
if [ -z "$GITHUB_ACTIONS" ]; then
  # The zip was created relative to release_package_manylinux_2_34, so it's in WORK_DIR
  # Move it to the build-output directory
  if [ -f "$WORK_DIR/$TEMP_ZIP" ]; then
    mkdir -p "$WORK_DIR/build-output"
    mv "$WORK_DIR/$TEMP_ZIP" "$WORK_DIR/build-output/$TEMP_ZIP"
    OUTPUT_ZIP="$WORK_DIR/build-output/$TEMP_ZIP"
  elif [ -f "$TEMP_ZIP" ]; then
    # File is in current directory
    mkdir -p build-output
    mv "$TEMP_ZIP" "build-output/$TEMP_ZIP"
    OUTPUT_ZIP="build-output/$TEMP_ZIP"
  fi
fi

# Set output for GitHub Actions
if [ -n "$GITHUB_ACTIONS" ] && [ -n "$GITHUB_OUTPUT" ]; then
  echo "asset_path=$OUTPUT_ZIP" >> "$GITHUB_OUTPUT"
fi

echo "Package created: $OUTPUT_ZIP"