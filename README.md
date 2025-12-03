# On-Prem Docs MCP Server

Run a **fully on-prem documentation assistant** that you can access from **Continue (VS Code + CLI)**. The MCP server performs semantic search and returns relevant document chunks, which Continue's LLM then synthesizes into answers.

With this setup you get these capabilities:

- **Ask questions** in Continue about your local PDF/DOCX/Markdown docs (no upload to third-party services)
- **Semantic search** over your docs using LlamaIndex
- **Source citations** for each answer (which chunks/files were used)
- **Local-only data flow**: docs and index stay on disk; only embeddings are generated locally
- **No duplicate LLM calls**: The MCP server only does retrieval; Continue's LLM synthesizes the answer

## Features

- Indexes PDF, DOCX, and Markdown documents from a local directory
- Uses LlamaIndex for semantic search and retrieval
- Exposes an MCP server with `retrieve_docs` tool that returns raw document chunks
- **No LLM required in the MCP server** - it only does retrieval; your existing Continue LLM handles answer synthesis
- Persistent index storage (no re-indexing on restart)

## Prerequisites

- Python 3.9+
- Documents in PDF, DOCX, or Markdown format

The intended deployment is **on-prem**:

- Your PDFs/DOCX/Markdown files live on your machines.
- Indexes are stored locally on disk.
- Answer generation is done by Continue's LLM (which you already have configured).
- The MCP server only does retrieval - no LLM needed!

## Installation

1. Clone this repository and navigate to the project directory:
```bash
git clone git@github.com:Chetic/on-prem-docs-mcp.git
cd on-prem-docs-mcp
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

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

Edit `.env` file with your settings:

- `DATA_DIR`: Directory containing PDF/DOCX/Markdown files (default: `./data`)
- `STORAGE_DIR`: Directory for index storage (default: `./storage`)
- `EMB_MODEL_NAME`: Embedding model name used by LlamaIndex (default: `BAAI/bge-small-en-v1.5`). Any embedding model supported by LlamaIndex can be used; BGE-small is just a good default from Hugging Face.
- `SIMILARITY_TOP_K`: Number of document chunks to retrieve per query (default: `5`)

**Note:** The MCP server does **not** need any LLM configuration (API base, model, API key). It only performs semantic search and returns raw chunks. Continue's LLM (which you already have configured) will synthesize the answer from these chunks.

## Usage

### 1. Prepare Documents

Place your PDF, DOCX, and Markdown files in the `data/` directory:
```bash
mkdir -p data
# Copy your PDF/DOCX/Markdown files to data/
```

### 2. Build the Index

Run the ingestion script to build the index:
```bash
python ingest.py
```

This will:
- Scan the `data/` directory for PDF, DOCX, and Markdown files
- Parse and chunk the documents
- Generate embeddings
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
   - macOS/Linux: `~/.continue/mcpServers/on-prem-docs-mcp.yaml`
   - Windows: `%USERPROFILE%\\.continue\\mcpServers\\on-prem-docs-mcp.yaml`
3. Add an MCP server entry in YAML:

```yaml
name: On-Prem Docs MCP
version: 1.0.0
schema: v1
mcpServers:
  - name: on-prem-docs-mcp
    command: python
    args:
      - path/to/on-prem-docs-mcp/mcp_server.py
    env:
      STORAGE_DIR: path/to/on-prem-docs-mcp/storage
```

**Note:** No LLM configuration needed! The MCP server only does retrieval. Continue's LLM handles answer synthesis.

Replace `path/to/on-prem-docs-mcp` with the path where you cloned the repository. After saving, reload VS Code; the `on-prem-docs-mcp` tools (including `retrieve_docs`) should appear in Continue's tool list.

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

## Testing

For concise testing instructions on how to run the Python test scripts, see `TESTING.md`.

## Project Structure

```
.
├── ingest.py          # Document ingestion and index building
├── mcp_server.py      # MCP server implementation
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variable template
├── .gitignore         # Git ignore rules
├── README.md          # This file
├── data/              # Input documents (PDF/DOCX/Markdown)
└── storage/           # Persistent index storage
```

## Troubleshooting

### Index Not Found Error

If you see "Storage directory does not exist", run `python ingest.py` first.

### Retrieval Errors

- Verify the index exists: Check that `STORAGE_DIR` contains the index files
- Run `python ingest.py` if the index doesn't exist
- Check that embedding model downloads completed successfully

### Embedding Model Download

The first run will download the embedding model from Hugging Face. This may take a few minutes.

## Future Enhancements

- Additional document sources (Confluence, Notion, etc.)
- Vector database integration (pgvector, Qdrant)
- Authentication and multi-tenancy
- Incremental indexing
- Query result caching
- Additional MCP tools (search_docs, list_docs)

## License

MIT

