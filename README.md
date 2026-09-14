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
- Incremental indexing — only reprocesses new or changed files, and never writes inside an indexed directory
- Revision-aware indexing — files that are revisions of one document (`Spec_v2.pdf`, `.../Rev B/Spec.docx`) are grouped and only the newest is indexed; review-comment copies are skipped
- Heading-aware results with source links back to the original file, spread across documents instead of one document's every chunk
- Date filtering and recency boosting
- Files indexed from a mounted network share can be presented at the share's own location (an `smb://` URI plus the Windows UNC path)
- Serves MCP over stdio to a local client, or over streamable-http behind a reverse proxy of your own (oauth2-proxy + KeyCloak walkthrough included)
- Optional Confluence and Jira integrations (supports Cloud and Server/Data Center)

## Installation

### Option A: Install from PyPI (Recommended)

Requires Python 3.11 or later. Models are downloaded automatically on first run (~250MB). The first run may appear to pause while models download — this is normal.

```bash
pip install chunksilo
```

Confluence and Jira support is included by default — just provide a config file
to enable them. (`pip install chunksilo[confluence,jira]` still works as an alias
for backward compatibility.)

Then:
1. **Create** a config file at `~/.config/chunksilo/config.yaml` (see [Configuration](#configuration))
2. **Build** the index: `chunksilo --build-index`
3. **Configure** your MCP client (see [docs/mcp-clients.md](docs/mcp-clients.md))

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
5. **Configure** your MCP client (see [docs/mcp-clients.md](docs/mcp-clients.md))

## Configuration

ChunkSilo reads one file, `config.yaml`, taken from `--config`, else the
`CHUNKSILO_CONFIG` environment variable, else `./config.yaml`, else
`~/.config/chunksilo/config.yaml`. Every setting is optional and
`chunksilo --dump-defaults` prints them all with their defaults.

```yaml
indexing:
  directories:
    - "./data"
    - "/mnt/docs"

retrieval:
  rerank_top_k: 5

storage:
  storage_dir: "./storage"
  model_cache_dir: "./models"
```

The full reference — per-directory filters, revision handling, retrieval
tuning, Confluence and Jira, share locations, the HTTP transport, and what
changed for users upgrading from 2.x — is in
[docs/configuration.md](docs/configuration.md).

## The `search_docs` tool

`search_docs` answers with the files whose names matched and the individual
chunks that carry the answer, each located by a URI, the page or line it came
from, and its heading path:

```json
{
  "matched_files": [
    { "uri": "file:///docs/database-configuration.docx", "unc": null, "score": 0.8432 }
  ],
  "num_matched_files": 1,
  "chunks": [
    {
      "text": "To configure the database connection, set the DATABASE_URL environment variable...",
      "score": 0.912,
      "location": {
        "uri": "file:///docs/setup-guide.pdf",
        "unc": null,
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

See [docs/tools.md](docs/tools.md) for every field, including how files on a
mounted network share are presented.

## Documentation

- [docs/configuration.md](docs/configuration.md) — the config file, where it is discovered, and a reference for every setting.
- [docs/cli.md](docs/cli.md) — `chunksilo` and `chunksilo-mcp` options, including `--check-files` for explaining why a file is or is not indexed.
- [docs/tools.md](docs/tools.md) — the `search_docs` result shape.
- [docs/mcp-clients.md](docs/mcp-clients.md) — client setup for Claude Code, Claude Desktop, Cline, Roo Code and opencode, over stdio or HTTP.
- [docs/reverse-proxy.md](docs/reverse-proxy.md) — sharing an instance behind an authenticating reverse proxy, with oauth2-proxy and KeyCloak group-based access.
- [docs/troubleshooting.md](docs/troubleshooting.md) — indexing surprises, network mounts, offline mode and more.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
