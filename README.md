# On-Prem Docs MCP Server

Run a **fully on-prem documentation assistant** that you can access from **Continue (VS Code + CLI)**. The MCP server performs semantic search and returns relevant document chunks, which Continue's LLM then synthesizes into answers.

With this setup you get these capabilities:

- **Ask questions** in Continue about your local PDF/DOCX/Markdown/TXT docs (no upload to third-party services)
- **Semantic search** over your docs using LlamaIndex
- **Source citations** for each answer (which chunks/files were used)
- **Local-only data flow**: docs and index stay on disk; only embeddings are generated locally
- **No duplicate LLM calls**: The MCP server only does retrieval; Continue's LLM synthesizes the answer

## Features

- Indexes PDF, DOCX, Markdown, and TXT documents from a local directory
- Uses LlamaIndex for semantic search and retrieval
- **CPU-optimized embedding stack** using FastEmbed (no PyTorch required for embeddings) with a lightweight CPU reranker
- Exposes an MCP server with `retrieve_docs` tool that returns raw document chunks
- **No LLM required in the MCP server** - it only does retrieval; your existing Continue LLM handles answer synthesis
- Persistent index storage (no re-indexing on restart)

## Prerequisites

- Python 3.11+
- Documents in PDF, DOCX, Markdown, or TXT format

The intended deployment is **on-prem**:

- Your PDFs/DOCX/Markdown/TXT files live on your machines.
- Indexes are stored locally on disk.
- Answer generation is done by Continue's LLM (which you already have configured).
- The MCP server only does retrieval - no LLM needed!

## Installation

1. Clone this repository and navigate to the project directory:
```bash
git clone git@github.com:Chetic/opd-mcp.git
cd opd-mcp
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `.env` file with your settings:

- `DATA_DIR`: Directory containing PDF/DOCX/Markdown/TXT files (default: `./data`)
- `STORAGE_DIR`: Directory for index storage (default: `./storage`)
- `RETRIEVAL_MODEL_CACHE_DIR`: Shared cache directory for the embedding and reranking models (default: `./models`). Include this directory in release artifacts so deployments do not need to download either model.
- `RETRIEVAL_EMBED_MODEL_NAME`: Embedding model name used for the vector search stage (default: `BAAI/bge-small-en-v1.5`). The default is a quantized ONNX BGE-small model that is already vendored in `./models` for offline use.
- `RETRIEVAL_EMBED_TOP_K`: Number of vector search candidates to pass from the embedding stage into the reranker (default: `10`).
- `RETRIEVAL_RERANK_MODEL_NAME`: Cross-encoder reranking model used after the embedding step (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`).
- `RETRIEVAL_RERANK_TOP_K`: Final number of reranked chunks returned by the `retrieve_docs` tool (default: `5`).
- `OFFLINE`: For the MCP server, set to `1` (default) to enforce offline mode and prevent HuggingFace libraries from making network requests. Set to `0` to allow network access if needed.

**Note:** The MCP server does **not** need any LLM configuration (API base, model, API key). It only performs semantic search and returns raw chunks. Continue's LLM (which you already have configured) will synthesize the answer from these chunks.

## Usage

### 1. Prepare Documents

Place your PDF, DOCX, Markdown, and TXT files in the `data/` directory:
```bash
mkdir -p data
# Copy your PDF/DOCX/Markdown/TXT files to data/
```

### 2. Download the retrieval models for offline use

The server runs fully offline once the embedding and reranking models are cached locally. To vendor the
models into the repository (so they can be included in a release artifact), run:

```bash
python ingest.py --download-models
```

This stores the FastEmbed embedding model and the cross-encoder reranker in `./models/` (configurable via `RETRIEVAL_MODEL_CACHE_DIR`).
Include this directory in your release package so deployments never need internet
access. The manual GitHub release workflow automatically downloads both models into the
packaged `models/` directory so the published ZIP works offline without extra steps.

The ingestion script and MCP server use `huggingface_hub`'s `snapshot_download` with
`local_files_only=True` to locate cached models. Embeddings are passed directly to
FastEmbed via the `specific_model_path` parameter, and the reranker loads from the same
cache directory. This completely bypasses FastEmbed's download step and keeps the
cross-encoder fully offline, enabling true offline operation.

### 3. Build the Index

Run the ingestion script to build the index (uses the offline model cache):
```bash
python ingest.py
```

This will:
- Scan the `data/` directory for PDF, DOCX, Markdown, and TXT files
- Parse and chunk the documents
- Generate embeddings using the locally cached model
- Build and persist the vector index to `storage/`

### 3. Configure Continue's LLM

The MCP server doesn't need its own LLM - it only does retrieval. Make sure Continue is configured with your preferred LLM (Ollama, OpenAI, or any other provider you use). Continue's LLM will synthesize answers from the document chunks returned by the MCP server.

### 4. Run the MCP Server

The MCP server can be run directly for testing:
```bash
python mcp_server.py
```

In production, the MCP server is typically started by an MCP client (e.g., Continue).

### 5. Configure MCP Client

Add this server to your MCP client configuration. Below are examples for **Continue**.

#### Continue (VS Code Extension)

1. Install the **Continue** extension from the VS Code Marketplace.
2. Create (or edit) a YAML MCP server config file, for example:
   - macOS/Linux: `~/.continue/mcpServers/opd-mcp.yaml`
   - Windows: `%USERPROFILE%\\.continue\\mcpServers\\opd-mcp.yaml`
3. Add an MCP server entry in YAML:

```yaml
name: On-Prem Docs MCP
version: 1.0.0
schema: v1
mcpServers:
  - name: opd-mcp
    command: python
    args:
      - path/to/opd-mcp/mcp_server.py
    env:
      STORAGE_DIR: path/to/opd-mcp/storage
```

**Note:** No LLM configuration needed! The MCP server only does retrieval. Continue's LLM handles answer synthesis.

Replace `path/to/opd-mcp` with the path where you cloned the repository. After saving, reload VS Code; the `opd-mcp` tools (including `retrieve_docs`) should appear in Continue's tool list.

#### Continue CLI

1. Install the CLI:

```bash
npm install -g @continuedev/cli
```

2. Ensure the same `mcpServers` configuration is present in your Continue config directory (the CLI reads the same YAML config under `~/.continue`, including files in `~/.continue/mcpServers/`).
3. Start an interactive session that can use the MCP tools:

```bash
cn chat
```

From within the CLI chat, you can ask the model to call the `retrieve_docs` tool exposed by this server.

### Response Fields and Citations

The `retrieve_docs` tool returns raw document chunks with metadata so the client LLM can answer questions and cite sources. Each chunk includes:

- `text`: Full chunk text (not truncated)
- `score`: Retrieval score for the chunk
- `metadata`: Original metadata from LlamaIndex (including file path/page when available)
- `citation`: Human-friendly citation string derived from the metadata (e.g., `docs/guide.pdf (page 3, section "Introduction")`)
- `location`: Structured location details that the client can surface in answers, including file path, page (when available), and section/heading information. This mirrors the `file`, `page`, `heading`, and `heading_path` fields returned by the MCP server so clients can consistently display where a chunk came from.

Responses also include a top-level `citations` array that lists unique citations used across all returned chunks. This makes it easy for clients to display or reference the sources alongside generated answers and ensures the model can explicitly cite which portion of each document it used.

## Testing

For concise testing instructions on how to run the Python test scripts, see `test/TESTING.md`.

## Commit messages

Please use **conventional commits without a scope** when contributing (for example: `feat: add progress bar`, `fix: address ingestion bug`, `chore: update docs`). Clear, scope-free messages keep release notes accurate and easy to generate.

## Project Structure

```
.
├── ingest.py          # Document ingestion and index building
├── mcp_server.py      # MCP server implementation
├── requirements.txt   # Python dependencies
├── .env       # Environment variable template
├── .gitignore         # Git ignore rules
├── README.md          # This file
├── data/              # Input documents (PDF/DOCX/Markdown/TXT)
└── storage/           # Persistent index storage
```

## Troubleshooting

### Index Not Found Error

If you see "Storage directory does not exist", run `python ingest.py` first.

### Retrieval Errors

- Verify the index exists: Check that `STORAGE_DIR` contains the index files
- Run `python ingest.py` if the index doesn't exist
- Check that retrieval model downloads (embedding + reranker) completed successfully

### Retrieval Model Download

The first run will download the embedding model and the reranker. FastEmbed automatically downloads and caches the embedding model on first use; the reranker is cached via the Hugging Face Hub. This may take a few minutes, but subsequent runs will use the cached copies.

For offline deployments and release builds, run `python ingest.py --download-models` ahead of time and include the resulting `models/` directory in the release artifact so both models are available without internet access.

### Offline Mode

When running with `--offline` flag or in an offline environment, the script automatically sets HuggingFace environment variables (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`) to prevent any network requests. The MCP server also defaults to offline mode to ensure no network access is attempted.

**Note:** Even with offline mode enabled, FastEmbed may attempt to validate the model from HuggingFace before using the cached version. This is a known limitation of FastEmbed's download logic. The script will verify the cache exists before attempting to initialize the model, but if FastEmbed still tries to download, you may see error messages. The download attempts will fail gracefully in offline mode, and FastEmbed should eventually use the cached model if it's properly cached.

If you see network connection errors even in offline mode, verify that:
- The models are fully cached in `RETRIEVAL_MODEL_CACHE_DIR` (default: `./models`)
- You're using the `--offline` flag when running `ingest.py`
- The `OFFLINE` environment variable is set to `1` (or not set, which defaults to offline mode)
- The cached model directory exists at `models/models--qdrant--bge-small-en-v1.5-onnx-q/` (or equivalent for your model)

## Future Enhancements

- Additional document sources (Confluence, Notion, etc.)
- Vector database integration (pgvector, Qdrant)
- Authentication and multi-tenancy
- Incremental indexing
- Query result caching
- Additional MCP tools (search_docs, list_docs)

## License

MIT

