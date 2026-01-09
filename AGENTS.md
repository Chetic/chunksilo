# Agent Guidelines

## Commit Standards
- Use conventional commit messages **without a scope** (e.g., `feat: add progress bar`, `chore: update docs`).
- Keep future commits descriptive so release notes remain accurate.

## Testing Requirements

**Before committing any changes, run the test suite:**
```bash
cd test && ./run_tests.sh
```

Or equivalently:
```bash
python3 -m pytest test/ -v --ignore=test/test_large_scale.py
```

**Requirements:**
- Python 3.11 is required for running tests
- All tests must pass before submitting changes
- New functionality should include appropriate unit or integration tests
- Tests are in the `test/` directory and use the pytest framework

**Test Files:**
- `test_chunk_location.py` - Unit tests for location field generation
- `test_heading_path_integration.py` - Integration tests for heading extraction
- `test_incremental_ingest.py` - Tests for incremental indexing
- `test_retrieval_only.py` - Tests for index loading and retrieval
- `test_system.py` - End-to-end system tests
- `test_large_scale.py` - RAG metrics evaluation (runs separately in CI)

## Package Guidelines
- **Test files must not affect the release package**: All test-related files are in the `test/` directory and are excluded from the release zip file. The release package is a standalone, offline-ready MCP server that users can unpack and run without any test dependencies or online connections.
- **Air-gapped Connectivity Exception**: While the server is designed for air-gapped environments, it is assumed that a local Confluence instance is reachable. Runtime retrieval from this specific local service is permitted.
