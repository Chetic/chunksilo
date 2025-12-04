## Testing On-Prem Docs MCP Server

This guide shows how to test the **retrieval-only MCP server** using the included Python test scripts.

### 1. Basic functionality (ingestion + index loading)

Verify ingestion and retrieval work using only embeddings and vector search (no LLM needed):

```bash
source venv/bin/activate
python test_retrieval_only.py
```

This will:
- Ingest documents from `DATA_DIR` (default `./data`)
- Build and persist the index into `STORAGE_DIR` (default `./storage`)
- Perform a retrieval-only test over the index

### 2. End-to-end retrieval test (MCP server logic)

To exercise the same retrieval logic that the MCP server uses (`retrieve_docs`), run the corresponding pytest:

```bash
source venv/bin/activate
pytest test_system.py
```

This will:
- Ensure the index exists (or rebuild it)
- Call into the MCP server logic (via `retrieve_docs`) to retrieve chunks
- Print sample queries and information about the retrieved chunks (count, scores, previews)

You can also run the full test suite (including ingestion and system tests) with:

```bash
source venv/bin/activate
pytest
```

### 3. Manual MCP server test

You can also run the MCP server directly:

```bash
source venv/bin/activate
python mcp_server.py
```

The server speaks MCP over stdio and is normally launched by an MCP client (such as Continue). Running it manually is useful for debugging, but you’ll need an MCP-aware client to actually send tool calls.


