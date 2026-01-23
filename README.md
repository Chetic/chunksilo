# ChunkSilo MCP Server

Local semantic search for PDF, DOCX, DOC, Markdown, and TXT files. The MCP server efficiently retrieves chunks of text with sources and the LLM can either answer questions or go to the document and find more information.

## Features

- **Privacy First**: Fully local retrieval (FastEmbed + lightweight reranker).
- **Smart Indexing**: Persistent local index with incremental updates.
- **Source Links**: MCP `search_docs` tool returns resource links for each source document, displayed as clickable links in supported MCP clients.
- **Dual Retrieval**: Returns both semantically relevant chunks and BM25 filename matches separately, so filename lookups don't get buried by semantic reranking.

## Quick Installation

Download the latest release package from the [Releases page](https://github.com/Chetic/chunksilo/releases).

1. **Download** the `chunksilo-vX.Y.Z-manylinux_2_34_x86_64.tar.gz` file
2. **Extract** and install:

```bash
tar -xzf chunksilo-vX.Y.Z-manylinux_2_34_x86_64.tar.gz
cd chunksilo
./setup.sh
```

3. **Edit** `config.yaml` to set your document directories
4. **Build** the index: `./venv/bin/python index.py`
5. **Configure** your MCP client (see [MCP Client Configuration](#mcp-client-configuration))

## Configuration

ChunkSilo uses a single configuration file: `config.yaml`

### Configuration File

Edit `config.yaml` to configure your settings:

```yaml
# Indexing settings - used by index.py when building the search index
indexing:
  directories:
    - "./data"
    - "/mnt/nfs/shared-docs"
    - path: "/mnt/samba/engineering"
      include: ["**/*.pdf", "**/*.md"]
      exclude: ["**/archive/**"]
  chunk_size: 1600
  chunk_overlap: 200

# Retrieval settings - used by chunksilo.py when searching
retrieval:
  embed_top_k: 20
  rerank_top_k: 5
  score_threshold: 0.1
  offline: true

# Confluence integration (optional)
confluence:
  url: "https://confluence.example.com"
  username: "your-username"
  api_token: "your-api-token"

# Storage paths (usually don't need to change)
storage:
  storage_dir: "./storage"
  model_cache_dir: "./models"
```

All settings are optional and have sensible defaults.

### Configuration Reference

#### Indexing Settings (used by index.py)

| Setting | Default | Description |
| :--- | :--- | :--- |
| `indexing.directories` | `["./data"]` | List of directories to index (strings or objects) |
| `indexing.chunk_size` | `1600` | Maximum size of text chunks |
| `indexing.chunk_overlap` | `200` | Overlap between adjacent chunks |

**Per-directory options** (when using object format):

| Option | Default | Description |
| :--- | :--- | :--- |
| `path` | (required) | Directory path to index |
| `include` | `["**/*.pdf", "**/*.md", "**/*.txt", "**/*.docx", "**/*.doc"]` | Glob patterns for files to include |
| `exclude` | `[]` | Glob patterns for files to exclude |
| `recursive` | `true` | Whether to recurse into subdirectories |
| `enabled` | `true` | Whether to index this directory |

#### Retrieval Settings (used by chunksilo.py)

| Setting | Default | Description |
| :--- | :--- | :--- |
| `retrieval.embed_model_name` | `BAAI/bge-small-en-v1.5` | Embedding model for vector search |
| `retrieval.embed_top_k` | `20` | Candidates from vector search before reranking |
| `retrieval.rerank_model_name` | `ms-marco-MiniLM-L-12-v2` | Reranker model |
| `retrieval.rerank_top_k` | `5` | Final results after reranking |
| `retrieval.rerank_candidates` | `100` | Maximum candidates sent to reranker |
| `retrieval.score_threshold` | `0.1` | Minimum score (0.0-1.0) for results |
| `retrieval.recency_boost` | `0.3` | Recency boost weight (0.0-1.0) |
| `retrieval.recency_half_life_days` | `365` | Days until recency boost halves |
| `retrieval.bm25_similarity_top_k` | `10` | Files returned by BM25 filename search |
| `retrieval.offline` | `true` | Prevent ML library network requests |

#### Confluence Settings (optional)

| Setting | Default | Description |
| :--- | :--- | :--- |
| `confluence.url` | `""` | Confluence base URL (empty = disabled) |
| `confluence.username` | `""` | Confluence username |
| `confluence.api_token` | `""` | Confluence API token |
| `confluence.timeout` | `10.0` | Request timeout in seconds |
| `confluence.max_results` | `30` | Maximum results per search |

#### SSL Settings (optional)

| Setting | Default | Description |
| :--- | :--- | :--- |
| `ssl.ca_bundle_path` | `""` | Path to custom CA bundle file |

#### Storage Settings

| Setting | Default | Description |
| :--- | :--- | :--- |
| `storage.storage_dir` | `./storage` | Directory for vector index and state |
| `storage.model_cache_dir` | `./models` | Directory for model cache |

## MCP Client Configuration

Configure your MCP client to run ChunkSilo. Below are examples for common clients.

### Claude Desktop / Generic MCP Client

Add to your MCP client's configuration file:

```json
{
  "mcpServers": {
    "chunksilo": {
      "command": "/path/to/chunksilo/venv/bin/python",
      "args": ["chunksilo.py"],
      "cwd": "/path/to/chunksilo"
    }
  }
}
```

### Cline (VS Code Extension)

Add to `cline_mcp_settings.json` (typically in `~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/`):

```json
{
  "mcpServers": {
    "chunksilo": {
      "command": "/path/to/chunksilo/venv/bin/python",
      "args": ["chunksilo.py"],
      "cwd": "/path/to/chunksilo",
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

### Roo Code (VS Code Extension)

Add to `mcp_settings.json` (typically in `~/.config/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/`):

```json
{
  "mcpServers": {
    "chunksilo": {
      "command": "/path/to/chunksilo/venv/bin/python",
      "args": ["chunksilo.py"],
      "cwd": "/path/to/chunksilo"
    }
  }
}
```

## Troubleshooting

- **Index missing**: Run `./venv/bin/python index.py` in the install directory.
- **Retrieval errors**: Check paths in your MCP client configuration.
- **Offline mode**: The release package includes models and sets `offline: true` by default. Set `retrieval.offline: false` in `config.yaml` if you need network access.
- **Confluence Integration**: Set `confluence.url`, `confluence.username`, and `confluence.api_token` in `config.yaml` to enable Confluence search.
- **Custom CA Bundle**: Set `ssl.ca_bundle_path` in `config.yaml` for custom certificates.
- **Network mounts**: Unavailable directories are skipped with a warning; indexing continues with available directories.
- **Legacy .doc files**: Requires LibreOffice to be installed for automatic conversion to .docx. If LibreOffice is not found, .doc files are skipped with a warning. Full heading extraction is supported.

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
