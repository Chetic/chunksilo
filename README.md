# On-Prem Docs MCP Server

Fully local semantic search for your PDF, DOCX, Markdown, and TXT files. The MCP server only retrieves chunks; your LLM (via Continue, Cline, or Roo Code) does the answering. No data leaves your machine.

## Features

- **Privacy First**: Fully local retrieval (FastEmbed + lightweight reranker).
- **Universal Installer**: Single self-contained script for installation and configuration.
- **Multi-Tool Support**: Auto-configuration for Cline, Roo Code, and Continue.
- **Smart Indexing**: Persistent local index with incremental updates.
- **Citation Support**: MCP `retrieve_docs` tool returns citation metadata for verifiable answers.

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

- **Settings**: Adjust `DATA_DIR` (documents), `STORAGE_DIR` (index), and `RETRIEVAL_MODEL_CACHE_DIR` directly in the generated config if needed, or modify `universal_config.json` before running the installer.
- **Documents**: Place your documents in the `DATA_DIR` (default: `data/` inside the install location).
- **Indexing**: The server runs `ingest.py` automatically, or you can run it manually:
  ```bash
  cd <install_dir>
  source venv/bin/activate
  python ingest.py
  ```

## Troubleshooting

- **Index missing**: Run `python ingest.py` in the install directory.
- **Retrieval errors**: Check paths in your tool's MCP config file.
- **Offline mode**: The installer includes models. Ensure `OFFLINE=1` is set in the environment or `.env` file (automatically handled by the installer).
- **Confluence Integration**: To enable Confluence search, set `CONFLUENCE_URL`, `CONFLUENCE_USERNAME`, and `CONFLUENCE_API_TOKEN` environment variables in your MCP client configuration (e.g., `cline_mcp_settings.json`, `config.json` for Continue, etc.).

## License

MIT
