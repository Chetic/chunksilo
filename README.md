<p align="center">
  <img src="https://raw.githubusercontent.com/Chetic/chunksilo/main/chunksilo.png" alt="ChunkSilo Logo" width="500">
</p>

<p align="center">
  <img src="demo/demo.gif" alt="ChunkSilo terminal demo" width="720">
</p>

# ChunkSilo MCP Server

ChunkSilo is like a local Google for your documents. It uses semantic search — matching by meaning rather than exact keywords — so your LLM can find relevant information across all your files even when the wording differs from your query. Point it at your PDFs, Word docs, Markdown, and text files, and it builds a fully searchable index locally on your machine.

- Runs entirely on your machine — no servers, no infrastructure
- Semantic search + keyword filename matching across PDF, DOCX, DOC, Markdown, and TXT
- Incremental indexing — only reprocesses new or changed files
- Heading-aware results with source links back to the original file
- Date filtering and recency boosting
- Optional Confluence and Jira integrations (supports Cloud and Server/Data Center)

### Example `search_docs` output

```json
{
  "matched_files": [
    { "uri": "file:///docs/database-configuration.docx", "score": 0.8432 }
  ],
  "num_matched_files": 1,
  "chunks": [
    {
      "text": "To configure the database connection, set the DATABASE_URL environment variable...",
      "score": 0.912,
      "location": {
        "uri": "file:///docs/setup-guide.pdf",
        "page": 12,
        "line": null,
        "heading_path": ["Getting Started", "Configuration", "Database"]
      }
    }
  ],
  "num_chunks": 1,
  "query": "how to configure the database",
  "retrieval_time": "0.42s"
}
```

## Installation

### Option A: Install from PyPI (Recommended)

Requires Python 3.11 or later. Models are downloaded automatically on first run (~250MB). The first run may appear to pause while models download — this is normal.

```bash
pip install chunksilo

# Or with Confluence and Jira support:
pip install chunksilo[confluence,jira]
```

Then:
1. **Create** a config file at `~/.config/chunksilo/config.yaml` (see [Configuration](#configuration))
2. **Build** the index: `chunksilo --build-index`
3. **Configure** your MCP client (see [MCP Client Configuration](#mcp-client-configuration))

### Option B: Offline Bundle

A self-contained package with pre-downloaded models, ideal for air-gapped environments or systems without Python installed.

Download from the [Releases page](https://github.com/Chetic/chunksilo/releases):

1. **Download** the `chunksilo-vX.Y.Z-manylinux_2_34_x86_64.tar.gz` file
2. **Extract** and install:

```bash
tar -xzf chunksilo-vX.Y.Z-manylinux_2_34_x86_64.tar.gz
cd chunksilo
./setup.sh
```

3. **Edit** `config.yaml` to set your document directories
4. **Build** the index: `./venv/bin/chunksilo --build-index`
5. **Configure** your MCP client (see [MCP Client Configuration](#mcp-client-configuration))

## Configuration

ChunkSilo uses a single configuration file: `config.yaml`

### Configuration File

Edit `config.yaml` to configure your settings:

```yaml
# Indexing settings - used by chunksilo --build-index
indexing:
  directories:
    - "./data"
    - "/mnt/nfs/shared-docs"
    - path: "/mnt/samba/engineering"
      include: ["**/*.pdf", "**/*.md"]
      exclude: ["**/archive/**"]
  chunk_size: 1600
  chunk_overlap: 200

# Retrieval settings - used when searching
retrieval:
  embed_top_k: 20
  rerank_top_k: 5
  score_threshold: 0.1

# Confluence integration (optional) - supports Cloud and Server/Data Center
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

> **Tip:** Run `chunksilo --dump-defaults` to see all available options with their default values.

#### Indexing Settings

| Setting | Description |
| :--- | :--- |
| `indexing.directories` | List of directories to index (strings or objects) |
| `indexing.chunk_size` | Maximum size of text chunks |
| `indexing.chunk_overlap` | Overlap between adjacent chunks |

**Per-directory options** (when using object format):

| Option | Description |
| :--- | :--- |
| `path` | Directory path to index (required) |
| `include` | Glob patterns for files to include |
| `exclude` | Glob patterns for files to exclude |
| `recursive` | Whether to recurse into subdirectories |
| `enabled` | Whether to index this directory |

**Project-wide directory defaults** — set once instead of repeating per directory:

| Option | Description |
| :--- | :--- |
| `indexing.defaults.include` | Default include patterns for all directories |
| `indexing.defaults.exclude` | Default exclude patterns for all directories |
| `indexing.defaults.recursive` | Default recursive setting for all directories |

**Advanced indexing options** — performance tuning, timeouts, and logging:

| Setting | Description |
| :--- | :--- |
| `indexing.parallel_workers` | Number of threads for parallel file loading |
| `indexing.enable_parallel_loading` | Enable/disable parallel file loading |
| `indexing.enable_adaptive_batching` | Enable memory-aware adaptive batch sizing |
| `indexing.max_memory_mb` | Memory budget (MB) for adaptive batch sizing |
| `indexing.checkpoint_interval_files` | Files processed between index checkpoints |
| `indexing.checkpoint_interval_seconds` | Seconds between index checkpoints |
| `indexing.timeout.enabled` | Enable per-file processing timeout |
| `indexing.timeout.per_file_seconds` | Timeout in seconds for processing each file |
| `indexing.timeout.doc_conversion_seconds` | Timeout in seconds for .doc to .docx conversion |
| `indexing.timeout.heartbeat_interval_seconds` | Interval (seconds) between progress animation updates during file processing |
| `indexing.logging.log_slow_files` | Warn when files take unusually long to process |
| `indexing.logging.slow_file_threshold_seconds` | Seconds before a file is considered slow |

#### Retrieval Settings

| Setting | Description |
| :--- | :--- |
| `retrieval.embed_model_name` | Embedding model for vector search |
| `retrieval.embed_top_k` | Candidates from vector search before reranking |
| `retrieval.rerank_model_name` | Reranker model |
| `retrieval.rerank_top_k` | Final results after reranking |
| `retrieval.rerank_candidates` | Maximum candidates sent to reranker |
| `retrieval.score_threshold` | Minimum score (0.0-1.0) for results |
| `retrieval.recency_boost` | Recency boost weight (0.0-1.0) |
| `retrieval.recency_half_life_days` | Days until recency boost halves |
| `retrieval.bm25_similarity_top_k` | Files returned by BM25 filename search |
| `retrieval.offline` | Prevent ML library network requests |

#### Confluence Settings (optional)

> **Note:** Confluence integration requires the optional dependency. Install with: `pip install chunksilo[confluence]`

| Setting | Description |
| :--- | :--- |
| `confluence.url` | Confluence base URL (empty = disabled) |
| `confluence.username` | Confluence username |
| `confluence.api_token` | Confluence API token (Cloud) or Personal Access Token (Server/Data Center) |
| `confluence.timeout` | Request timeout in seconds |
| `confluence.max_results` | Maximum results per search |

**Creating a Confluence API Token:**
1. Log into Confluence
2. Go to Account Settings > Security > API Tokens (for Cloud) or User Profile > Personal Access Tokens (for Server/Data Center)
3. Click "Create API Token" or "Create Token"
4. Copy the token and add it to your config

#### Jira Settings (optional)

> **Note:** Jira integration requires the optional dependency. Install with: `pip install chunksilo[jira]`

| Setting | Description |
| :--- | :--- |
| `jira.url` | Jira base URL (empty = disabled) |
| `jira.username` | Jira username/email |
| `jira.api_token` | Jira API token |
| `jira.timeout` | Request timeout in seconds |
| `jira.max_results` | Maximum results per search |
| `jira.projects` | Project keys to search (empty = all) |
| `jira.include_comments` | Include issue comments in search |
| `jira.include_custom_fields` | Include custom fields in search |

**Creating a Jira API Token:**
1. Log into Jira
2. Go to Account Settings > Security > API Tokens
3. Click "Create API Token"
4. Copy the token and add it to your config

#### SSL Settings (optional)

| Setting | Description |
| :--- | :--- |
| `ssl.ca_bundle_path` | Path to custom CA bundle file |

#### Storage Settings

| Setting | Description |
| :--- | :--- |
| `storage.storage_dir` | Directory for vector index and state |
| `storage.model_cache_dir` | Directory for model cache |

## CLI Usage

```bash
chunksilo --build-index                # Build or update the search index
chunksilo "your search query"          # Search for documents
chunksilo "report" --date-from 2024-01-01 --date-to 2024-03-31  # Date filtering
chunksilo --dump-defaults              # Print all config options with defaults
```

### CLI Options

| Option | Description |
| :--- | :--- |
| `query` | Search query text (positional argument) |
| `--build-index` | Build or update the search index with step-by-step progress output, then exit |
| `--download-models` | Download required ML models, then exit |
| `--dump-defaults` | Print all default configuration values as YAML, then exit |
| `--date-from` | Start date filter (YYYY-MM-DD format, inclusive) |
| `--date-to` | End date filter (YYYY-MM-DD format, inclusive) |
| `--json` | Output results as JSON instead of formatted text |
| `-v, --verbose` | Show diagnostic messages (model loading, search stats) |
| `--config` | Path to config.yaml (overrides auto-discovery) |
| `CHUNKSILO_CONFIG` | Environment variable alternative to `--config` |

## MCP Client Configuration

Configure your MCP client to run ChunkSilo. Below are examples for common clients.

> **Note:** For PyPI installs, use `chunksilo-mcp` directly. For offline bundles, use the full path `/path/to/chunksilo/venv/bin/chunksilo-mcp`. You can find the PyPI-installed binary location with `which chunksilo-mcp`.

### Claude Code

Add chunksilo as an MCP server using the CLI:

**PyPI install:**
```bash
claude mcp add chunksilo --scope user -- chunksilo-mcp --config ~/.config/chunksilo/config.yaml
```

**Offline bundle:**
```bash
claude mcp add chunksilo --scope user -- /path/to/chunksilo/venv/bin/chunksilo-mcp --config /path/to/chunksilo/config.yaml
```

Verify it's connected:

```bash
claude mcp list
```

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

**PyPI install:**
```json
{
  "mcpServers": {
    "chunksilo": {
      "command": "chunksilo-mcp",
      "args": ["--config", "/path/to/config.yaml"]
    }
  }
}
```

**Offline bundle:**
```json
{
  "mcpServers": {
    "chunksilo": {
      "command": "/path/to/chunksilo/venv/bin/chunksilo-mcp",
      "args": ["--config", "/path/to/chunksilo/config.yaml"]
    }
  }
}
```

### Cline (VS Code Extension)

Add to `cline_mcp_settings.json` (typically in `~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/`):

**PyPI install:**
```json
{
  "mcpServers": {
    "chunksilo": {
      "command": "chunksilo-mcp",
      "args": ["--config", "/path/to/config.yaml"],
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

**Offline bundle:**
```json
{
  "mcpServers": {
    "chunksilo": {
      "command": "/path/to/chunksilo/venv/bin/chunksilo-mcp",
      "args": ["--config", "/path/to/chunksilo/config.yaml"],
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

### Roo Code (VS Code Extension)

Add to `mcp_settings.json` (typically in `~/.config/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/`):

**PyPI install:**
```json
{
  "mcpServers": {
    "chunksilo": {
      "command": "chunksilo-mcp",
      "args": ["--config", "/path/to/config.yaml"]
    }
  }
}
```

**Offline bundle:**
```json
{
  "mcpServers": {
    "chunksilo": {
      "command": "/path/to/chunksilo/venv/bin/chunksilo-mcp",
      "args": ["--config", "/path/to/chunksilo/config.yaml"]
    }
  }
}
```

## Troubleshooting

- **Index missing**: Run `chunksilo --build-index` (PyPI install) or `./venv/bin/chunksilo --build-index` (offline bundle).
- **Retrieval errors**: Check paths in your MCP client configuration.
- **Offline mode**: PyPI installs default to `offline: false` (models auto-download). The offline bundle includes pre-downloaded models and sets `offline: true`. Set `retrieval.offline: true` in `config.yaml` to prevent network calls after initial model download.
- **Confluence Integration**: Install with `pip install chunksilo[confluence]`, then set `confluence.url`, `confluence.username`, and `confluence.api_token` in `config.yaml`.
- **Jira Integration**: Install with `pip install chunksilo[jira]`, then set `jira.url`, `jira.username`, and `jira.api_token` in `config.yaml`. Optionally configure `jira.projects` to restrict search to specific project keys.
- **Custom CA Bundle**: Set `ssl.ca_bundle_path` in `config.yaml` for custom certificates.
- **Network mounts**: Unavailable directories are skipped with a warning; indexing continues with available directories.
- **Legacy .doc files**: Requires LibreOffice to be installed for automatic conversion to .docx. If LibreOffice is not found, .doc files are skipped with a warning. Full heading extraction is supported.

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
