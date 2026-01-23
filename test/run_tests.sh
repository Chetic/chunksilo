#!/bin/bash
# Run the full functional test suite locally
# Usage: ./run_tests.sh [pytest args]
#
# Examples:
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh -v           # Run with verbose output
#   ./run_tests.sh -k "chunk"   # Run tests matching "chunk"
#   ./run_tests.sh --tb=short   # Short traceback format
#
# Requirements:
#   - Python 3.11
#   - Dependencies from requirements.txt and test/requirements.txt

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Find Python 3.11 with dependencies installed
# Prioritize virtual environment over system Python
if [[ -f "$PROJECT_ROOT/.venv311/bin/python" ]]; then
    PYTHON="$PROJECT_ROOT/.venv311/bin/python"
elif [[ -f "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
elif command -v python3.11 &> /dev/null; then
    PYTHON=python3.11
else
    echo "Error: Python 3.11 is required but not found."
    echo "Please create a virtual environment:"
    echo "  python3.11 -m venv .venv311"
    echo "  .venv311/bin/pip install -r requirements.txt -r test/requirements.txt"
    exit 1
fi

# Verify Python version
PYTHON_VERSION=$($PYTHON --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
if [[ "$PYTHON_VERSION" != "3.11" ]]; then
    echo "Error: Python 3.11 is required. Found: Python $PYTHON_VERSION"
    exit 1
fi

cd "$PROJECT_ROOT"

# Run all tests except rag_metrics (which requires downloads and takes longer)
$PYTHON -m pytest test/ -v --ignore=test/test_rag_metrics.py "$@"
