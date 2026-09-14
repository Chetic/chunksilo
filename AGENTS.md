# Agent Guidelines

## Commit Standards
- Use conventional commit messages (e.g., `feat: add progress bar`, `chore: update docs`).
- Keep future commits descriptive so release notes remain accurate.

## Testing Requirements

**Before committing any changes, run the test suite:**
```bash
test/run_tests.sh
```
(equivalent to `python3 -m pytest test/ -v --ignore=test/test_rag_metrics.py --ignore=test/test_indexing_benchmark.py`), and `ruff check src/`.

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
- The README is a short overview; reference material lives in `docs/` (one file per topic, each starting with `Part of [ChunkSilo](../README.md).`). Keep `config.yaml`, `chunksilo --dump-defaults` and `docs/configuration.md` in agreement.

## Configuration Invariants
- **Config activation contract**: `index.py` reads config at import time via `load_config()`, and an explicit-path `load_config(path)` becomes the process-wide active config that `cfgload.get()` answers from. `cli.main`, `run_server` and `build_index` therefore call `load_config(path)` before importing/using `.search`/`.index`; keep that ordering if refactoring startup. Tests get a fresh active config per test from the autouse fixture in `test/conftest.py`.
- **Removed options warn, they do not silently vanish**: when an option is dropped from the schema, add it to `cfgload._REMOVED_KEYS` with its replacement.

## Indexing Invariants
- **Indexing never writes inside an indexed directory.** All scratch space (the
  .doc conversion work dirs, the LibreOffice profile), the index, and the logs
  live under `storage.storage_dir`; sources are opened read-only and .doc files
  are copied out before LibreOffice ever sees them. `test/test_never_mutates.py`
  enforces this with a byte-for-byte snapshot - keep it passing.
- **A file missing from a scan is only deleted if the scan could see where it lives.** `LocalFileSystemSource` records unreadable files, subdirectories it could not list (`os.walk` skips those silently) and incomplete walks on `ScanStatus`; `build_index` subtracts those from the deletion set. Pruning on a failed lookup drops the file's chunks and state row, so the next run re-indexes it — on a laggy network mount that becomes a permanent re-index loop.
- **Every attempted file gets a state row unless it failed.** `load_files_parallel` returns a `LoadOutcome` per file: `empty` (extracted no text) is a final answer and is persisted with no doc ids; `failed` gets no row so it is retried. Dropping empty results makes those files look new on every run for ever. A loader signals failure by raising (`FileLoadError`, `FileProcessingTimeoutError`, `OSError`), never by returning an empty list: a timeout or a failed `.doc` conversion returned as `[]` would be persisted as "holds no text" and never retried.
- **`--check-files` runs the indexer's own scan.** `filecheck.run_check` drives the same `iter_candidates` / `_match_decision` / revision partitioning as `build_index`; a change to the scan that is not visible through those APIs will make the dry run lie.

## Search Invariants
- **Result formatting must not touch the filesystem.** Result paths often live on a network mount where a stat plus a readlink per component dominates the whole query; `_normalize_result_path` and the share-location mapping (`shareuri.py`) work from the stored string and the configuration alone. Resolving would also rewrite a path into the mount's internal form, which no application can open. `test/test_chunk_location.py` pins this.
- **The lazy loaders are double-checked-locked.** `_ensure_embed_model`, `_ensure_reranker`, `_ensure_bm25_retriever` and `load_llamaindex_index` are process-wide; over the HTTP transport tool calls run concurrently in a thread pool with no session affinity, so without the locks each first caller cold-starts its own copy. `run_server` calls `search.warm_up` before serving HTTP.
- **The HTTP transport authenticates nobody.** Access control belongs to a reverse proxy in front of the server; do not add credentials, tokens or per-user filtering to the transport itself. `server.public_url` / `allowed_hosts` only widen the SDK's Host/Origin allow-list, `server.authorization_servers` only publishes discovery metadata, and a forwarded `X-Forwarded-User` is logged, never acted on.
