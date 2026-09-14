# Configuration

Part of [ChunkSilo](../README.md).

ChunkSilo uses a single configuration file: `config.yaml`

## Configuration File

The file is found in this order: the `--config PATH` argument, the
`CHUNKSILO_CONFIG` environment variable, `./config.yaml`, then
`~/.config/chunksilo/config.yaml`. A path given with `--config` governs the
whole process — every setting is read from that file, not only the ones the
command line happens to mention.

Edit `config.yaml` to configure your settings:

```yaml
# Indexing settings - used by chunksilo --build-index
indexing:
  directories:
    - "./data"
    - "/mnt/nfs/shared-docs"
    - path: "/mnt/docs/engineering"
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

All settings are optional and have sensible defaults. An empty value
(`chunk_size:` with nothing after it) keeps the default rather than setting the
option to nothing.

## Configuration Reference

> **Tip:** Run `chunksilo --dump-defaults` to see all available options with their default values.

### Indexing Settings

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
| `indexing.defaults.case_sensitive` | Whether include/exclude globs are case-sensitive (default: false) |

Use `chunksilo --check-files` to see what these filters do to a tree without
building the index (see [cli.md](cli.md#testing-the-file-filters---check-files)).

**Advanced indexing options** — performance tuning, timeouts, and diagnostics:

| Setting | Description |
| :--- | :--- |
| `indexing.parallel_workers` | Threads for parallel file loading (1 = serial) |
| `indexing.batch_size` | Embedding batch size upper bound (adapts down under memory pressure) |
| `indexing.checkpoint_interval_files` | Files processed between index checkpoints |
| `indexing.checkpoint_interval_seconds` | Seconds between index checkpoints |
| `indexing.per_file_seconds` | Give up on a single file after this long (0 disables) |
| `indexing.scan_item_seconds` | Timeout for stat/hash/walk operations while scanning |
| `indexing.slow_file_threshold_seconds` | Warn when one file takes longer than this |

**Revision handling** — copes with ad-hoc document version control, where
revisions live in revision-named directories (`.../PB3/Spec.docx`,
`.../Rev B/Spec.docx`) or carry a filename token (`Spec PB3.docx`,
`Spec_v2.pdf`):

| Setting | Description |
| :--- | :--- |
| `indexing.versioning.enabled` | Group revisions of one document and index only the newest (default: true) |
| `indexing.versioning.doc_id_patterns` | Regexes tried against the full path; files sharing an extracted document ID form one group. First capture group (or the whole match) is the ID. Default: empty |
| `indexing.versioning.review_copy_patterns` | Filename markers of review-comment copies, which are never indexed. Setting this replaces the defaults wholesale; `[]` disables detection |
| `indexing.versioning.extra_revision_tokens` | Extra revision-token regex fragments (case-insensitive) added to the built-in grammar |
| `indexing.versioning.index_superseded` | Index older revisions too, stamped and demoted below current documents in results (default: false) |

How it works:

- Files are grouped by document ID when a `doc_id_patterns` regex matches
  (for example `['\b(DOC-\d{5})\b']` groups every file whose path contains
  the same `DOC-12345`); otherwise by their normalized name and location —
  revision tokens are stripped from the filename and revision-named directory
  components from the path, case-insensitively. The file extension stays part
  of the identity (`Spec.pdf` and `Spec.docx` are separate documents), except
  that `.doc` and `.docx` count as one.
- The built-in token grammar recognizes preliminary/released letters
  (`PA1`, `pb3`, a trailing single uppercase letter like `Spec B.docx`),
  `v`-numbers (`v2`, `V1.0`), and `rev`/`revision`/`ver`/`version`/`utg`/`utgåva`
  markers. Whole directory components additionally match bare letters and
  one-or-two-digit numbers (`B/`, `2/`). Trailing letters after words like
  *Appendix*, *Annex*, *Bilaga* are left alone.
- **The newest revision is the one with the newest file modification time.**
  Revision labels are never ordered — with no consistent labelling convention,
  mtime is the only signal that generalizes.
- Default review-copy markers: `review`, `comment(s)`, `kommentar(er)`,
  `granskning`, `copy`, `kopia` (word-bounded, case-insensitive, filename
  only). Note these also match documents legitimately named after the
  activity ("Literature Review.docx") — `chunksilo --build-index --verbose`
  logs every skipped file, and the pattern list is yours to narrow.
- Skipped files are removed from the index on the next `--build-index` and
  come back automatically if they become the newest again (for example when
  the latest revision is deleted). Set `indexing.versioning.enabled: false`
  to restore the old index-everything behavior.

### Retrieval Settings

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
| `retrieval.max_chunks_per_doc` | Cap on one document's chunks in the final results, so several documents surface instead of one document's every chunk; freed slots backfill from the next-best chunks (0 disables) |
| `retrieval.offline` | Prevent ML library network requests |

The score threshold and phrase filters run over the whole reranked candidate
list before the cut to `rerank_top_k`, and raising `retrieval.embed_top_k`
widens the pool that `max_chunks_per_doc` diversifies over, at some reranking
cost.

### Confluence Settings (optional)

> **Note:** Confluence support is installed by default; just set the values below to enable it.

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

### Jira Settings (optional)

> **Note:** Jira support is installed by default; just set the values below to enable it.

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

### SSL Settings (optional)

| Setting | Description |
| :--- | :--- |
| `ssl.ca_bundle_path` | CA bundle for environments with TLS interception or a private CA: used for model downloads and the Confluence/Jira APIs |

### Storage Settings

| Setting | Description |
| :--- | :--- |
| `storage.storage_dir` | Directory for the vector index, the ingestion state, scratch space for `.doc` conversion, and the MCP server log (`mcp.log`) |
| `storage.model_cache_dir` | Directory for model cache |

Indexing never writes inside an indexed directory: everything it produces lands
under `storage.storage_dir`.

### Server Settings

| Setting | Description |
| :--- | :--- |
| `server.transport` | `stdio` (default; the MCP client starts the server itself) or `streamable-http` (listen for remote clients) |
| `server.host` | Bind address for the HTTP transport (default `127.0.0.1`) |
| `server.port` | Port for the HTTP transport (default `8400`) |
| `server.public_url` | The URL a reverse proxy serves this instance on, e.g. `https://search.example.com`. Required behind a proxy: the transport rejects any `Host` header it does not expect with `421`, and this adds the public hostname (and origin) to what it accepts |
| `server.allowed_hosts` | Further `Host` header values to accept (`host` or `host:port`; `host:*` for any port) |
| `server.allowed_origins` | Further `Origin` values to accept (`scheme://host[:port]`; `scheme://host:*` for any port) |
| `server.authorization_servers` | Issuer URLs (e.g. a KeyCloak realm) published as RFC 9728 protected-resource metadata at `/.well-known/oauth-protected-resource`, so MCP clients can discover where to obtain a token. Requires `public_url`. ChunkSilo never verifies tokens itself |

The HTTP transport performs **no authentication of its own**: anyone who can
reach the port can search everything in the index. Keep the loopback bind, or
put a reverse proxy that terminates TLS and authenticates users in front of
the server before exposing it — see [reverse-proxy.md](reverse-proxy.md) for
an oauth2-proxy and KeyCloak setup. `chunksilo-mcp --transport streamable-http`
overrides the config for one run, the HTTP server warms up the index and
models before it accepts its first request, and `GET /health` answers once it
is listening.

### Share Locations (optional)

| Setting | Description |
| :--- | :--- |
| `shares` | List of `{prefix, unc}` entries. Files under a local mount `prefix` are presented at the share's own location, `unc`, as an `smb://` URI plus the Windows UNC path (see [tools.md](tools.md#share-locations)) |

```yaml
shares:
  - prefix: "/mnt/docs"      # local mount point as listed in indexing.directories
    unc: "//nas/docs"        # the same tree as clients reach it
```

Prefixes are matched component-wise on the normalized path (so `/mnt/docs`
does not cover `/mnt/docs-archive`, and `..` cannot escape into another
share), and the longest matching prefix wins. Entries with a relative prefix
or an `unc` that is not `//server/share...` are ignored with a warning. Files
under no prefix keep their `file://` URI.

## Upgrading from 2.x

Some indexing options were flattened or removed. A config file that still sets
one of them is loaded (the option is ignored) and a warning names its
replacement:

| Removed | Use instead |
| :--- | :--- |
| `indexing.timeout.enabled`, `indexing.timeout.per_file_seconds` | `indexing.per_file_seconds` (0 disables) |
| `indexing.timeout.scan_item_seconds` | `indexing.scan_item_seconds` |
| `indexing.timeout.doc_conversion_seconds`, `indexing.timeout.heartbeat_interval_seconds` | internal now |
| `indexing.logging.slow_file_threshold_seconds` | `indexing.slow_file_threshold_seconds` |
| `indexing.logging.log_slow_files` | never read; removed |
| `indexing.enable_parallel_loading` | `indexing.parallel_workers: 1` for serial loading |
| `indexing.enable_adaptive_batching`, `indexing.max_memory_mb` | `indexing.batch_size` is an upper bound; sizing adapts automatically |
| `retrieval.bm25_similarity_top_k` | never read; removed |

Other behaviour changes: result `file://` URIs are now percent-encoded (a
space is `%20`), each result carries a `unc` field (`null` unless `shares`
covers the file), and by default only the newest revision of a document is
indexed — set `indexing.versioning.enabled: false` for the old
index-everything behaviour. No index rebuild is required.
