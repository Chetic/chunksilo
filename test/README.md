# Test Suite

This directory contains all test-related files for ChunkSilo.

## Structure

- `test_retrieval_only.py` - Basic retrieval-only tests (no LLM needed)
- `test_system.py` - End-to-end system tests using pytest
- `test_rag_metrics.py` - RAG metrics test suite with evaluation metrics
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
OFFLINE=0 python test_rag_metrics.py
```

## Important Notes

- **Test files are excluded from the release package**: The release workflow only includes production files (`chunksilo.py`, `index.py`, `requirements.txt`, etc.). Test files remain in the repository for development but are not packaged for end users.
- **Tests require online access**: The RAG metrics test suite downloads documents from the web, so run with `OFFLINE=0`.
- **Test artifacts are gitignored**: Generated test data, storage, and results are in `.gitignore` to keep the repository clean.

