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
PACKAGE_ROOT="release_package_manylinux_2_34/chunksilo-${VERSION}"
mkdir -p "$PACKAGE_ROOT"

# Copy common files and models (including hidden files/directories)
# Enable dotglob to match hidden files with *
shopt -s dotglob
cp -r release_common/* "$PACKAGE_ROOT/"
shopt -u dotglob

# Set up paths
WORK_DIR="/workspace"
[ ! -d "$WORK_DIR" ] && WORK_DIR="$PROJECT_ROOT"
CONSTRAINTS_FILE="$WORK_DIR/scripts/manylinux_2_34-constraints.txt"

# Verify constraints file exists
if [ ! -f "$CONSTRAINTS_FILE" ]; then
  echo "Error: Constraints file not found at $CONSTRAINTS_FILE" >&2
  exit 1
fi

# Download dependencies for manylinux_2_34 compatibility
python3.11 -m pip install --upgrade pip

mkdir -p "$PACKAGE_ROOT/dependencies"

echo "Downloading dependencies with manylinux_2_34 constraints..."
python3.11 -m pip download \
  -r "$PACKAGE_ROOT/requirements.txt" \
  llama-index-readers-confluence \
  jira \
  -c "$CONSTRAINTS_FILE" \
  -d "$PACKAGE_ROOT/dependencies" \
  --no-cache-dir \
  --disable-pip-version-check

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

# Generate third-party license report from bundled dependencies
echo "Generating third-party license report..."
python3.11 -m venv /tmp/license-venv
/tmp/license-venv/bin/pip install --quiet --no-index --find-links "$PACKAGE_ROOT/dependencies" \
  -r "$PACKAGE_ROOT/requirements.txt" llama-index-readers-confluence jira
/tmp/license-venv/bin/pip install --quiet pip-licenses

# Validate that installed package count is reasonable vs downloaded wheels
INSTALLED_COUNT=$(/tmp/license-venv/bin/pip list --format=columns | tail -n +3 | wc -l)
WHEEL_COUNT=$(find "$PACKAGE_ROOT/dependencies" -type f \( -name "*.whl" -o -name "*.tar.gz" \) | wc -l)
echo "Installed packages: $INSTALLED_COUNT, Downloaded wheels: $WHEEL_COUNT"
if [ "$INSTALLED_COUNT" -lt "$((WHEEL_COUNT / 2))" ]; then
  echo "Error: Only $INSTALLED_COUNT of $WHEEL_COUNT packages installed -- license report would be incomplete" >&2
  rm -rf /tmp/license-venv
  exit 1
fi

/tmp/license-venv/bin/pip-licenses --format=plain-vertical --with-license-file --no-license-path \
  --output-file="$PACKAGE_ROOT/THIRD-PARTY-LICENSES.txt"
rm -rf /tmp/license-venv

if [ ! -s "$PACKAGE_ROOT/THIRD-PARTY-LICENSES.txt" ]; then
  echo "Error: THIRD-PARTY-LICENSES.txt is empty or missing" >&2
  exit 1
fi
echo "✓ Third-party license report generated"

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