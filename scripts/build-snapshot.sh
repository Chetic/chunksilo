#!/bin/bash
# build-snapshot.sh
# Simple script to build snapshot bundles for local offline testing
#
# This script produces the same offline bundle as the GitHub workflow,
# but with a snapshot version for testing purposes (not for release).
#
# Usage:
#   ./scripts/build-snapshot.sh              # Auto snapshot version
#   ./scripts/build-snapshot.sh --version test-v1  # Custom version

set -eo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_OUTPUT="$PROJECT_ROOT/build-output"

# Default values
VERSION=""
SHOW_HELP=false

# Colors for output (if terminal supports it)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    BOLD=''
    NC=''
fi

# Show help message
show_help() {
    cat <<EOF
${BOLD}build-snapshot.sh${NC} - Build offline bundle for local testing

${BOLD}USAGE:${NC}
    $0 [OPTIONS]

${BOLD}DESCRIPTION:${NC}
    Builds a ChunkSilo offline bundle for testing in isolated/offline environments.
    Uses the same build process as the GitHub workflow, but produces a snapshot
    version for testing (not an official release).

${BOLD}OPTIONS:${NC}
    --version VERSION    Use custom version string (default: snapshot-YYYYMMDD-HHMMSS)
    --help              Show this help message

${BOLD}EXAMPLES:${NC}
    # Build with auto-generated snapshot version
    $0

    # Build with custom test version
    $0 --version test-v1

${BOLD}OUTPUT:${NC}
    build-output/chunksilo-{VERSION}-manylinux_2_34_x86_64.tar.gz

${BOLD}REQUIREMENTS:${NC}
    - Docker (running)
    - Python 3.11
    - ~5GB disk space for models and dependencies

${BOLD}VERIFICATION:${NC}
    After building, test the offline bundle:

    tar -xzf build-output/chunksilo-*-manylinux_2_34_x86_64.tar.gz
    cd chunksilo
    ./setup.sh
    ./venv/bin/chunksilo --help
EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --version)
                VERSION="$2"
                shift 2
                ;;
            --help|-h)
                SHOW_HELP=true
                shift
                ;;
            *)
                echo -e "${RED}Error: Unknown option: $1${NC}" >&2
                echo "Use --help for usage information." >&2
                exit 1
                ;;
        esac
    done
}

# Generate snapshot version if not provided
get_snapshot_version() {
    if [ -n "$VERSION" ]; then
        echo "$VERSION"
    else
        # Generate timestamp-based version: snapshot-YYYYMMDD-HHMMSS
        date +snapshot-%Y%m%d-%H%M%S
    fi
}

# Validate prerequisites before starting
validate_prerequisites() {
    local errors=0

    echo -e "${BLUE}Validating prerequisites...${NC}"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}✗ Docker is not installed${NC}" >&2
        echo "  Install from: https://docs.docker.com/engine/install/" >&2
        errors=$((errors + 1))
    elif ! docker info &> /dev/null 2>&1; then
        echo -e "${RED}✗ Docker daemon is not running${NC}" >&2
        echo "  Start Docker and try again" >&2
        errors=$((errors + 1))
    else
        echo -e "${GREEN}✓ Docker is available${NC}"
    fi

    # Check Python 3.11
    if ! command -v python3.11 &> /dev/null; then
        echo -e "${RED}✗ Python 3.11 is required but not found${NC}" >&2
        echo "  Install Python 3.11 and try again" >&2
        errors=$((errors + 1))
    else
        local py_version=$(python3.11 --version 2>&1 | cut -d' ' -f2)
        echo -e "${GREEN}✓ Python 3.11 is available${NC} ($py_version)"
    fi

    # Check git repository
    if ! git rev-parse --git-dir &> /dev/null 2>&1; then
        echo -e "${YELLOW}⚠ Not in a git repository${NC}" >&2
        echo "  This is not critical, but recommended" >&2
    fi

    # Check disk space (need ~5GB for models + dependencies)
    local available_kb=$(df -k "$PROJECT_ROOT" | tail -1 | awk '{print $4}')
    local required_kb=$((5 * 1024 * 1024))  # 5GB
    local available_gb=$((available_kb / 1024 / 1024))

    if [ "$available_kb" -lt "$required_kb" ]; then
        echo -e "${YELLOW}⚠ Low disk space: ${available_gb}GB available (need ~5GB)${NC}" >&2
    else
        echo -e "${GREEN}✓ Sufficient disk space${NC} (${available_gb}GB available)"
    fi

    echo ""

    if [ "$errors" -gt 0 ]; then
        echo -e "${RED}Prerequisites check failed with $errors error(s)${NC}" >&2
        return 1
    fi

    return 0
}

# Main execution
main() {
    # Parse arguments
    parse_arguments "$@"

    # Show help if requested
    if [ "$SHOW_HELP" = true ]; then
        show_help
        exit 0
    fi

    # Show banner
    echo ""
    echo -e "${BOLD}======================================"
    echo "ChunkSilo Snapshot Bundle Builder"
    echo -e "======================================${NC}"
    echo ""

    # Validate prerequisites
    validate_prerequisites || exit 1

    # Determine version
    VERSION=$(get_snapshot_version)
    echo -e "${BOLD}Building snapshot version:${NC} $VERSION"
    echo ""

    # Create output directory
    mkdir -p "$BUILD_OUTPUT"

    # Step 1: Prepare common files
    echo -e "${BLUE}Step 1: Preparing common files (models, source, config)...${NC}"
    if ! "$SCRIPT_DIR/prepare-common.sh" "$VERSION"; then
        echo -e "${RED}✗ Failed to prepare common files${NC}" >&2
        exit 1
    fi
    echo -e "${GREEN}✓ Common files prepared${NC}"
    echo ""

    # Step 2: Build manylinux_2_34 package
    echo -e "${BLUE}Step 2: Building manylinux_2_34 package (this may take several minutes)...${NC}"
    if ! "$SCRIPT_DIR/package-manylinux-2_34.sh" "$VERSION"; then
        echo -e "${RED}✗ Failed to build package${NC}" >&2
        exit 1
    fi
    echo -e "${GREEN}✓ Package built${NC}"
    echo ""

    # Validate output
    EXPECTED_OUTPUT="$BUILD_OUTPUT/chunksilo-${VERSION}-manylinux_2_34_x86_64.tar.gz"
    if [ ! -f "$EXPECTED_OUTPUT" ]; then
        echo -e "${RED}✗ Expected output not found: $EXPECTED_OUTPUT${NC}" >&2
        exit 1
    fi

    # Get file size
    local file_size=$(du -h "$EXPECTED_OUTPUT" | cut -f1)

    # Success report
    echo -e "${BOLD}======================================"
    echo "Build completed successfully!"
    echo -e "======================================${NC}"
    echo ""
    echo -e "${BOLD}Output:${NC}"
    echo -e "  ${GREEN}$EXPECTED_OUTPUT${NC}"
    echo -e "  Size: $file_size"
    echo ""
    echo -e "${BOLD}Next steps:${NC}"
    echo "  1. Test the offline bundle:"
    echo "     tar -xzf $EXPECTED_OUTPUT"
    echo "     cd chunksilo"
    echo "     ./setup.sh"
    echo "     ./venv/bin/chunksilo --help"
    echo ""
    echo "  2. Copy to offline environment for testing:"
    echo "     scp $EXPECTED_OUTPUT user@offline-host:~/"
    echo ""
    echo -e "${BOLD}Cleanup:${NC}"
    echo "  rm -rf release_common/ release_package_manylinux_2_34/ build-output/"
    echo ""
}

# Run main function
main "$@"
