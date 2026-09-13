# MCP Client Configuration

Part of [ChunkSilo](../README.md).

Configure your MCP client to run ChunkSilo. Below are examples for common clients.

> **Note:** Two shapes appear below. **PyPI install** is `pip install chunksilo`
> into a virtualenv you activate, which puts `chunksilo-mcp` on your `PATH` —
> locate it with `which chunksilo-mcp`. **Offline bundle** is the unpacked
> release tarball, which is not on any `PATH`, so give the full
> `/path/to/chunksilo/venv/bin/chunksilo-mcp`.

## Claude Code

Add chunksilo as an MCP server using the CLI:

**PyPI install** (`chunksilo-mcp` on your `PATH`):
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

## Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

**PyPI install** (`chunksilo-mcp` on your `PATH`):
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

## Cline (VS Code Extension)

Add to `cline_mcp_settings.json` (typically in `~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/`):

**PyPI install** (`chunksilo-mcp` on your `PATH`):
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

## Roo Code (VS Code Extension)

Add to `mcp_settings.json` (typically in `~/.config/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/`):

**PyPI install** (`chunksilo-mcp` on your `PATH`):
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

## opencode

Add an `mcp` entry to your opencode config — `~/.config/opencode/opencode.json` for
all projects, or `opencode.json` in a project root for just that one. Config files
are merged, with the project file taking precedence.

**PyPI install** (`chunksilo-mcp` on your `PATH`):
```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "chunksilo": {
      "type": "local",
      "command": ["chunksilo-mcp", "--config", "/home/you/.config/chunksilo/config.yaml"],
      "enabled": true
    }
  }
}
```

**Offline bundle** — use the absolute path to the bundle's binary:
```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "chunksilo": {
      "type": "local",
      "command": [
        "/path/to/chunksilo/venv/bin/chunksilo-mcp",
        "--config", "/path/to/chunksilo/config.yaml"
      ],
      "enabled": true
    }
  }
}
```

Note that `command` is an array — the executable and each argument are separate
elements, not one string. Start opencode and ask it something answerable from your
documents to confirm the `search_docs` tool is being called.

## Remote server over HTTP

With `server.transport: streamable-http` (or `chunksilo-mcp --transport
streamable-http`) the server listens on `server.host:server.port` — by default
`http://127.0.0.1:8400/mcp` — and any MCP client that speaks streamable HTTP
can connect to it. ChunkSilo authenticates nobody on this transport, so expose
it only behind a reverse proxy that does; the examples below use the loopback
address.

**Claude Code:**
```bash
claude mcp add --transport http chunksilo http://127.0.0.1:8400/mcp
```

**opencode:**
```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "chunksilo": {
      "type": "remote",
      "url": "http://127.0.0.1:8400/mcp",
      "enabled": true
    }
  }
}
```
