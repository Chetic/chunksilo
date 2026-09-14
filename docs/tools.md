# MCP Tools

Part of [ChunkSilo](../README.md).

ChunkSilo exposes one MCP tool, `search_docs`, which finds relevant passages
across everything that was indexed (plus live Confluence and Jira results when
those integrations are configured).

## Example `search_docs` output

```json
{
  "matched_files": [
    {
      "uri": "file:///docs/database-configuration.docx",
      "unc": null,
      "score": 0.8432
    }
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

- `matched_files` — files whose *names* matched the query, one entry per
  document (the newest revision of a document group), at most five.
- `chunks` — the passages that carry the answer, in rank order. At most
  `retrieval.max_chunks_per_doc` chunks come from any one document, so the
  list spans documents instead of one document's every chunk.
- `location.uri` — a percent-encoded `file://` URI for a local file, the page
  URL for a Confluence result, or the issue URL for a Jira result.
- `location.unc` — the Windows UNC path of a file on a mapped network share
  (see below); `null` otherwise.
- `location.page` / `location.line` — the page number (PDF, DOCX) or line
  number (Markdown, text) the chunk starts on.
- `location.heading_path` — the section hierarchy the chunk sits under.

Errors are reported as an `error` field with a generic message; the detail is
in the server log.

## Share locations

A file that lives under one of the `shares` prefixes in the configuration is
presented at its real network location — an `smb://` URI plus the Windows UNC
path — never at the local mount path, which means nothing to a client on
another machine:

```json
{
  "uri": "smb://nas/docs/setup-guide.pdf",
  "unc": "\\\\nas\\docs\\setup-guide.pdf",
  "page": 12,
  "line": null,
  "heading_path": ["Getting Started", "Configuration", "Database"]
}
```

The `smb://` form is percent-encoded like a URL; the UNC form is raw, because
Windows opens `\\server\share\...` verbatim. Anything not covered by a share
mapping (including everything in a single-machine setup, which typically
configures none) keeps a `file:///...` URI, which the same-machine client can
open directly. See [configuration.md](configuration.md#share-locations-optional).
