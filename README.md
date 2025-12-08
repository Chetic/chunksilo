# On-Prem Docs MCP Server

Fully local semantic search for your PDFs, DOCX, Markdown, and TXT files. The MCP server only retrieves chunks; Continue’s LLM does the answering. No data leaves your machine.

## Features

- CPU-only retrieval stack (FastEmbed + lightweight reranker)
- Sentence-aware chunking with configurable size/overlap
- Persistent local index; offline-friendly model cache
- MCP `retrieve_docs` tool with citations metadata

## Quickstart (from source)

```bash
git clone git@github.com:Chetic/opd-mcp.git
cd opd-mcp
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp env.template .env
```

Key `.env` / environment settings (defaults in parentheses):
- `DATA_DIR` (`./data`): where your docs live.
- `STORAGE_DIR` (`./storage`): index location (also holds `ingestion_state.db`).
- `RETRIEVAL_MODEL_CACHE_DIR` (`./models`): cache for embedding + reranker; bundle for offline.
- `RETRIEVAL_EMBED_MODEL_NAME` (`BAAI/bge-small-en-v1.5`): embedding model.
- `RETRIEVAL_RERANK_MODEL_NAME` (`ms-marco-MiniLM-L-12-v2`): reranker; `cross-encoder/ms-marco-MiniLM-L-6-v2` maps to L-12.
- `CHUNK_SIZE` (`512`): token chunk size.
- `CHUNK_OVERLAP` (`100`): token overlap.
- `OFFLINE` (`1`): set Hugging Face caches to offline-friendly values.

Prepare data and models (CLI flags supported):
```bash
mkdir -p data
python ingest.py --download-models   # optional; just cache models then exit
python ingest.py                     # builds index from DATA_DIR into STORAGE_DIR
python ingest.py --offline           # force offline use of cached models
```

Run the server:
```bash
python mcp_server.py
```

Configure Continue (VS Code or CLI):
- Copy `continue-config/mcpServers/opd-mcp.yaml` to `~/.continue/mcpServers/opd-mcp.yaml` (update paths).
- Continue handles the LLM; no LLM config needed here.

## Deployment from release ZIP (offline)

1) Download `opd-mcp-<version>-manylinux_2_28_x86_64.zip` (RHEL 8/older) or `...-manylinux_2_34_x86_64.zip` (RHEL 9/newer) from Releases.  
2) Extract and enter the directory.  
3) `cp env.template .env` and adjust paths.  
4) `python3.11 -m venv venv && source venv/bin/activate`  
5) `pip install --no-index --find-links dependencies -r requirements.txt`  
6) Put docs in `DATA_DIR`, then `python ingest.py`. Models are already in `models/` for offline use.  
7) Copy/update the sample Continue config from `continue-config/` and start the server with `python mcp_server.py`.

## Response fields

`retrieve_docs` returns chunks with `text`, `score`, `metadata`, `citation`, and structured `location` (file, page, heading). A top-level `citations` array lists unique sources for easy display.

## Troubleshooting (quick)

- Index missing: run `python ingest.py`.
- Retrieval errors: confirm `STORAGE_DIR` exists and models are cached in `RETRIEVAL_MODEL_CACHE_DIR`.
- Offline runs: ensure `OFFLINE=1` or `--offline` and cached model directories exist under `./models`.

## Testing

See `test/TESTING.md` for running the Python test scripts.

## Commit messages

Use conventional commits without a scope (e.g., `feat: add progress bar`).

## Project structure

```
├── ingest.py          # Index build + model download
├── mcp_server.py      # MCP server
├── continue-config/   # Continue config examples
├── env.template       # .env template
├── models/            # Optional cached models
├── data/              # Your documents
└── storage/           # Built index
```

## License

MIT

