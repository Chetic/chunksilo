# Test Suite

This directory contains all test-related files for the On-Prem Docs MCP Server.

## Structure

- `test_retrieval_only.py` - Basic retrieval-only tests (no LLM needed)
- `test_system.py` - End-to-end system tests using pytest
- `test_large_scale.py` - Large-scale automated test suite with RAG evaluation metrics
- `create_test_doc.py` - Utility script to create test documents
- `TESTING.md` - Comprehensive testing documentation
- `requirements.txt` - Test-only dependencies (not included in release package)

## Installation

Install test dependencies:

```bash
pip install -r test/requirements.txt
```

## Running Tests

See `TESTING.md` for detailed instructions. All tests require `OFFLINE=0` to download test documents:

```bash
cd test
OFFLINE=0 python test_large_scale.py
```

## Important Notes

- **Test files are excluded from the release package**: The release workflow only includes production files (`mcp_server.py`, `ingest.py`, `requirements.txt`, etc.). Test files remain in the repository for development but are not packaged for end users.
- **Tests require online access**: The large-scale test suite downloads documents from the web, so run with `OFFLINE=0`.
- **Test artifacts are gitignored**: Generated test data, storage, and results are in `.gitignore` to keep the repository clean.

