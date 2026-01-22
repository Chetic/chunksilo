#!/bin/bash
# Script to build manylinux_2_34 package
# Works both locally (via Docker) and in CI (directly in container)

set -eo pipefail

VERSION="${1:-dev}"
CONTAINER_IMAGE="registry.access.redhat.com/ubi9/ubi:9.7"
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
    -e LOCAL_DOCKER=1 \
    "$CONTAINER_IMAGE" \
    bash /workspace/scripts/package-manylinux-2_34.sh "$VERSION"
  
  echo "Package built successfully: build-output/chunksilo-${VERSION}-manylinux_2_34_x86_64.tar.gz"
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
dnf install -y python3.11 python3.11-pip git which gcc gcc-c++ make

# Prepare manylinux_2_34 release package
PACKAGE_ROOT="release_package_manylinux_2_34/chunksilo"
mkdir -p "$PACKAGE_ROOT"

# Copy common files and models (including hidden files/directories)
# Enable dotglob to match hidden files with *
shopt -s dotglob
cp -r release_common/* "$PACKAGE_ROOT/"
shopt -u dotglob

# Copy RHEL 9.7 constraints file when available to ensure pillow <11 and
# llama-index compatibility. Fall back to minimal-constraints if missing.
WORK_DIR="/workspace"
[ ! -d "$WORK_DIR" ] && WORK_DIR="$PROJECT_ROOT"
RHEL97_CONSTRAINTS="$WORK_DIR/scripts/rhel9.7-constraints.txt"
if [ -f "$RHEL97_CONSTRAINTS" ]; then
  cp "$RHEL97_CONSTRAINTS" "$PACKAGE_ROOT/rhel9.7-constraints.txt"
  echo "Using RHEL 9.7 tested constraints file"
else
  echo "Warning: RHEL 9.7 constraints file not found at $RHEL97_CONSTRAINTS" >&2
  echo "Falling back to minimal-constraints.txt" >&2
fi

# Download dependencies for RHEL 9.7 compatibility
# We're running in UBI 9.7 container which uses glibc 2.34+, compatible with manylinux_2_34
# Strategy: Let pip automatically select compatible wheels for the current system
# This avoids dependency resolution issues from strict platform constraints
python3.11 -m pip install --upgrade pip

echo "Downloading dependencies for RHEL 9.7 (manylinux_2_34 compatible)..."

# Create dependencies directory
mkdir -p "$PACKAGE_ROOT/dependencies"

# Strategy: Download compatible wheels for the current system (RHEL 9.7)
# pip will automatically select wheels compatible with the current platform and Python version
# This is simpler and more reliable than specifying platform constraints
# We allow source distributions as fallback for packages without wheels

# Try 1: Download with RHEL 9.7 constraints (preferred for consistency)
echo "Attempting to download dependencies with RHEL 9.7 constraints..."
if python3.11 -m pip download \
  -r "$PACKAGE_ROOT/requirements.txt" \
  -c "$PACKAGE_ROOT/rhel9.7-constraints.txt" \
  -d "$PACKAGE_ROOT/dependencies" \
  --no-cache-dir \
  --disable-pip-version-check 2>&1; then
  echo "✓ Successfully downloaded dependencies with RHEL 9.7 constraints"
else
  echo "✗ Failed with RHEL 9.7 constraints, trying minimal constraints..."

  # Try 2: Download with minimal-constraints as a fallback
  if python3.11 -m pip download \
    -r "$PACKAGE_ROOT/requirements.txt" \
    -c "$PACKAGE_ROOT/minimal-constraints.txt" \
    -d "$PACKAGE_ROOT/dependencies" \
    --no-cache-dir \
    --disable-pip-version-check 2>&1; then
    echo "✓ Successfully downloaded dependencies with minimal constraints"
  else
    echo "✗ Failed with minimal constraints, trying without constraints..."

    # Try 3: Download without constraints (allows pip to resolve freely)
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
fi

# Show dependency inventory and fail fast if empty
DEP_COUNT=$(find "$PACKAGE_ROOT/dependencies" -type f | wc -l)
echo "Downloaded dependency files: $DEP_COUNT"
if [ "$DEP_COUNT" -eq 0 ]; then
  echo "Error: Dependencies directory is empty after download attempt" >&2
  exit 1
fi

# Verify dependencies were downloaded
if [ ! -d "$PACKAGE_ROOT/dependencies" ] || [ -z "$(ls -A "$PACKAGE_ROOT/dependencies" 2>/dev/null)" ]; then
  echo "Error: Dependencies directory is empty" >&2
  exit 1
fi

echo "✓ Dependencies downloaded successfully to $PACKAGE_ROOT/dependencies"

# Determine output path based on context
WORK_DIR="/workspace"
[ ! -d "$WORK_DIR" ] && WORK_DIR="$PROJECT_ROOT"

if [ -n "$GITHUB_ACTIONS" ] && [ -z "$LOCAL_DOCKER" ]; then
  # In actual CI, output to current directory
  OUTPUT_TAR="chunksilo-${VERSION}-manylinux_2_34_x86_64.tar.gz"
  TAR_PATH="../$OUTPUT_TAR"
  FINAL_OUTPUT="$OUTPUT_TAR"
else
  # Local Docker execution or when LOCAL_DOCKER is set, output to build-output directory
  OUTPUT_DIR="$WORK_DIR/build-output"
  mkdir -p "$OUTPUT_DIR"
  OUTPUT_TAR="$OUTPUT_DIR/chunksilo-${VERSION}-manylinux_2_34_x86_64.tar.gz"
  # Create tarball in current directory first, then move it
  TEMP_TAR="chunksilo-${VERSION}-manylinux_2_34_x86_64.tar.gz"
  TAR_PATH="../$TEMP_TAR"
  FINAL_OUTPUT="$OUTPUT_TAR"
fi

# Create tar.gz file (preserves file permissions including +x)
(cd "$WORK_DIR/release_package_manylinux_2_34" && tar -czvf "$TAR_PATH" .)

# Move to final location if needed (for local Docker execution)
if [ -n "$LOCAL_DOCKER" ] || [ -z "$GITHUB_ACTIONS" ]; then
  # The tarball was created relative to release_package_manylinux_2_34, so it's in WORK_DIR
  # Move it to the build-output directory
  if [ -f "$WORK_DIR/$TEMP_TAR" ]; then
    mkdir -p "$WORK_DIR/build-output"
    mv "$WORK_DIR/$TEMP_TAR" "$WORK_DIR/build-output/$TEMP_TAR"
    OUTPUT_TAR="$WORK_DIR/build-output/$TEMP_TAR"
  elif [ -f "$TEMP_TAR" ]; then
    # File is in current directory
    mkdir -p "$WORK_DIR/build-output"
    mv "$TEMP_TAR" "$WORK_DIR/build-output/$TEMP_TAR"
    OUTPUT_TAR="$WORK_DIR/build-output/$TEMP_TAR"
  fi
fi

# Set output for GitHub Actions
if [ -n "$GITHUB_ACTIONS" ] && [ -n "$GITHUB_OUTPUT" ]; then
  echo "asset_path=$OUTPUT_TAR" >> "$GITHUB_OUTPUT"
fi

echo "Package created: $OUTPUT_TAR"