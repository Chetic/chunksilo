# On-Prem Docs MCP Server

Fully local semantic search for your PDF, DOCX, Markdown, and TXT files. The MCP server only retrieves chunks; your LLM (via Continue, Cline, or Roo Code) does the answering. No data leaves your machine.

## Features

- **Privacy First**: Fully local retrieval (FastEmbed + lightweight reranker).
- **Universal Installer**: Single self-contained script for installation and configuration.
- **Multi-Tool Support**: Auto-configuration for Cline, Roo Code, and Continue.
- **Smart Indexing**: Persistent local index with incremental updates.
- **Source Links**: MCP `retrieve_docs` tool returns resource links for each source document, displayed as clickable links in supported MCP clients like Roo Code.

## Quick Installation (Recommended)

The easiest way to install is using the self-contained installer script from the [Releases page](https://github.com/Chetic/opd-mcp/releases).

1. **Download** the `opd-mcp-vX.Y.Z-installer.sh` file.
2. **Run** the installer:

```bash
chmod +x opd-mcp-installer.sh
./opd-mcp-installer.sh
```

All parameters are optional. The installer will ask you for any required information if it is not provided via flags.

### Installer Options

| Option | Description |
| :--- | :--- |
| `--tool <name>` | Target tool to configure: `cline`, `roo`, `continue`. |
| `--project [path]` | Configure for a specific project. Defaults to global if omitted. |
| `--editor <name>` | For global install: `code`, `cursor`, `windsurf`, `antigravity`, `vscodium`, etc. Auto-detects VS Code if it's the only available editor. |
| `--location <path>` | Install destination (defaults to `/data/opd-mcp`, `/localhome/opd-mcp`, or `~/opd-mcp`). |
| `--overwrite` | Force overwrite of existing files and configs. |

## Manual / Developer Installation

If you prefer to run from source:

1. Clone the repository.
2. Run `setup.sh` (which the installer wraps):
   ```bash
   ./setup.sh --tool <tool>
   ```

## Configuration

The installer generates tool-specific configurations from a single source of truth: `universal_config.json`.

- **Settings**: Adjust environment variables directly in the generated config if needed, or modify `universal_config.json` before running the installer.
- **Documents**: Place your documents in the `DATA_DIR` (default: `data/` inside the install location).
- **Indexing**: The server runs `ingest.py` automatically, or you can run it manually:
  ```bash
  cd <install_dir>
  source venv/bin/activate
  python ingest.py
  ```

### Environment Variables

Configure the MCP server by setting environment variables in your MCP client configuration (e.g., `cline_mcp_settings.json`, `config.json` for Continue, etc.).

#### Core Settings

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DATA_DIR` | `./data` | Directory containing PDF, DOCX, Markdown (.md), and TXT (.txt) files to index |
| `STORAGE_DIR` | `./storage` | Directory for storing the LlamaIndex vector index and ingestion state |
| `RETRIEVAL_MODEL_CACHE_DIR` | `./models` | Cache directory for embedding and reranking models |
| `OFFLINE` | `1` | Offline mode (1=enabled, 0=disabled). Prevents all network requests by ML libraries |

#### Retrieval Settings

| Variable | Default | Description |
| :--- | :--- | :--- |
| `RETRIEVAL_EMBED_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Hugging Face embedding model for vector search (stage 1) |
| `RETRIEVAL_EMBED_TOP_K` | `20` | Number of candidates retrieved from vector search before reranking |
| `RETRIEVAL_RERANK_MODEL_NAME` | `ms-marco-MiniLM-L-12-v2` | FlashRank reranker model for semantic reranking (stage 2) |
| `RETRIEVAL_RERANK_TOP_K` | `5` | Final number of results returned after reranking |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.1` | Minimum reranker score (0.0-1.0) for results. Set to 0.0 to disable filtering |
| `RETRIEVAL_RECENCY_BOOST` | `0.3` | Weight for recency boost (0.0=disabled, 1.0=recency dominates relevance) |
| `RETRIEVAL_RECENCY_HALF_LIFE_DAYS` | `365` | Days until a document's recency boost is halved (exponential decay) |

#### Text Chunking (ingest.py only)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `CHUNK_SIZE` | `512` | Maximum size of text chunks for indexing |
| `CHUNK_OVERLAP` | `100` | Number of overlapping characters between adjacent chunks |

#### Confluence Integration (Optional)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `CONFLUENCE_URL` | None | Confluence base URL (e.g., `https://confluence.example.com`). If unset, Confluence search is disabled |
| `CONFLUENCE_USERNAME` | None | Confluence username for authentication |
| `CONFLUENCE_API_TOKEN` | None | Confluence API token for authentication |
| `CONFLUENCE_TIMEOUT` | `10.0` | Timeout in seconds for Confluence search requests |
| `CONFLUENCE_MAX_RESULTS` | `30` | Maximum number of results to retrieve from Confluence |
| `CONFLUENCE_CLOUD` | Auto | Set to `true` for Confluence Cloud, `false` for Server/Data Center. Auto-detects based on URL if not set (*.atlassian.net = Cloud) |

#### SSL/TLS Settings (Optional)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `CA_BUNDLE_PATH` | None | Path to CA bundle file for custom certificates (e.g., self-signed or internal CA). Used for HTTPS connections to Confluence and other endpoints. Example: `/path/to/ca-bundle.crt` |

## Troubleshooting

- **Index missing**: Run `python ingest.py` in the install directory.
- **Retrieval errors**: Check paths in your tool's MCP config file.
- **Offline mode**: The installer includes models and sets `OFFLINE=1` automatically. If you need network access, set `OFFLINE=0` in your MCP client configuration.
- **Confluence Integration**: Set `CONFLUENCE_URL`, `CONFLUENCE_USERNAME`, and `CONFLUENCE_API_TOKEN` in your MCP client configuration to enable Confluence search.
- **Custom CA Bundle**: Set `CA_BUNDLE_PATH` to point to your CA bundle file if using custom certificates for HTTPS endpoints.

## License

MIT
