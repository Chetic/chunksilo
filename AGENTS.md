# Agent Guidelines

## Commit Standards
- Use conventional commit messages (e.g., `feat: add progress bar`, `chore: update docs`).
- Keep future commits descriptive so release notes remain accurate.

## Testing Requirements

**Before committing any changes, run the test suite:**
```bash
python3 -m pytest test/ -v --ignore=test/test_rag_metrics.py
```

**Requirements:**
- Python 3.11 is required for running tests
- All tests must pass before submitting changes
- New functionality should include appropriate unit or integration tests
- Tests are in the `test/` directory and use the pytest framework

## Package Guidelines
- **Test files must not affect the release package**: All test-related files are in the `test/` directory and are excluded from the release zip file. The release package is a standalone, offline-ready MCP server that users can unpack and run without any test dependencies or online connections.
- **Air-gapped Connectivity Exception**: While the server can operate in air-gapped environments, it is assumed that a local Confluence instance is reachable. Runtime retrieval from this specific local service is permitted.

## Documentation Guidelines
- **Do not claim absolute data privacy**: ChunkSilo runs search locally, but results are passed to the user's MCP client LLM, which may be cloud-hosted. Never state or imply that "no data leaves the network" — only state that ChunkSilo itself does not make external calls (when offline mode is enabled). The distinction is: ChunkSilo doesn't phone home, but the LLM client receiving results may.
