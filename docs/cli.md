# CLI Usage

Part of [ChunkSilo](../README.md).

```bash
chunksilo --build-index                # Build or update the search index
chunksilo "your search query"          # Search for documents
chunksilo "report" --date-from 2024-01-01 --date-to 2024-03-31  # Date filtering
chunksilo --dump-defaults              # Print all config options with defaults
chunksilo --list-files                 # List every indexed file
chunksilo --check-files                # Dry-run the file filters, report every verdict
chunksilo --check-files /path/to/file  # Explain why one file is or is not indexed
```

## CLI Options

| Option | Description |
| :--- | :--- |
| `query` | Search query text (positional argument) |
| `--build-index` | Build or update the search index with step-by-step progress output, then exit |
| `--download-models` | Download required ML models, then exit |
| `--dump-defaults` | Print all default configuration values as YAML, then exit |
| `--date-from` | Start date filter (YYYY-MM-DD format, inclusive) |
| `--date-to` | End date filter (YYYY-MM-DD format, inclusive) |
| `--json` | Output results as JSON instead of formatted text |
| `-v, --verbose` | Show diagnostic messages (model loading, search stats, every file the indexer skipped or reprocessed and why) |
| `--list-files` | List all indexed file paths, then exit |
| `--check-files [PATH]` | Dry-run the indexing file filters without building the index, then exit. Reports a verdict per file; with PATH, explains that one file (see below) |
| `--config` | Path to config.yaml (overrides auto-discovery) |
| `CHUNKSILO_CONFIG` | Environment variable alternative to `--config` |

## Testing the file filters (`--check-files`)

`--check-files` runs the exact scan the indexer uses — directory availability,
include/exclude globs, directory pruning, revision grouping, review-copy
detection — and reports what would happen, without touching the index, the
state database, or the models. Use it to answer "why is this file (not) being
indexed?" without re-indexing:

```text
$ chunksilo --check-files
State DB: ./storage/ingestion_state.db (present)

/srv/docs
  scanned (recursive)
  pruned dir    /srv/docs/.git  [by '**/.git/**']
  superseded    /srv/docs/Spec A.docx  [older revision; latest in group is /srv/docs/Spec B.docx]
  would index   /srv/docs/Spec B.docx  [matched include pattern '**/*.docx']  (indexed)
  review copy   /srv/docs/Spec review.docx  [review copy (filename matched '\breview\b'); never indexed]
  not included  /srv/docs/notes.xlsx  [no include pattern matched]

Summary: 1 would index, 1 superseded, 1 review copies, 1 not matching any include, 1 directories pruned
```

With a path argument it explains that one file, including its revision group:

```text
$ chunksilo --check-files "/srv/docs/Spec A.docx"
Path: /srv/docs/Spec A.docx
Verdict: superseded - older revision; latest in group is /srv/docs/Spec B.docx
Revision group: path:/srv/docs/spec.docx
  would index   /srv/docs/Spec B.docx
```

Exit codes make it scriptable: `0` when every enabled directory was scanned
completely (full mode) or the target file would be indexed (path mode); `1` on
a configuration error, an unavailable or incompletely scanned directory, or a
target that would be skipped. `--json` emits the full report as JSON — on
large trees, filter it with `jq` or `grep`.

## Server Options (`chunksilo-mcp`)

| Option | Description |
| :--- | :--- |
| `--config` | Path to config.yaml |
| `--transport` | `stdio` or `streamable-http`, overriding `server.transport` (see [configuration.md](configuration.md#server-settings)) |

The server writes its log to `<storage_dir>/mcp.log`, never to the current
directory.
