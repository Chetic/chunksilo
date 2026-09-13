#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
MCP server for querying documentation using RAG.
Returns raw document chunks for the calling LLM to synthesize.

Two transports:
- stdio (default): the MCP client starts this process itself; one user, one
  machine.
- streamable-http: listens on server.host:server.port for any MCP client that
  can reach it. This server performs NO authentication of its own - keep the
  loopback bind (the default) or put an authenticating reverse proxy in front
  of it before exposing the port.
"""
import argparse
import asyncio
import logging
import logging.handlers
import os
from pathlib import Path
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

# Log file configuration. The log lives in the storage directory - never the
# CWD, which on a mis-started service could be inside an indexed tree - and
# is private: it records what was searched.
LOG_FILE_NAME = "mcp.log"
LOG_MAX_SIZE_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5

TRANSPORTS = ("stdio", "streamable-http")

# Module-level state (set during initialization)
_server_config_path: Path | None = None
_mcp: FastMCP | None = None


class _PrivateRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler whose files are only ever mode 0600."""

    def _open(self):
        fd = os.open(
            self.baseFilename, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
        )
        # O_CREAT does not chmod a file that already exists, so be explicit.
        os.fchmod(fd, 0o600)
        return os.fdopen(fd, self.mode, encoding=self.encoding)


def _setup_logging(storage_dir: str | Path):
    """Configure logging for MCP server mode - file only, no stdout/stderr."""
    log_path = Path(storage_dir).expanduser() / LOG_FILE_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
        handlers=[
            _PrivateRotatingFileHandler(
                log_path,
                maxBytes=LOG_MAX_SIZE_BYTES,
                backupCount=LOG_BACKUP_COUNT,
                encoding="utf-8",
            ),
        ],
    )
    logging.getLogger("llama_index.readers.confluence").setLevel(logging.WARNING)


def _create_server(
    config: dict[str, Any] | None = None, transport: str = "stdio"
) -> FastMCP:
    """Create and configure the MCP server with tools.

    For ``streamable-http`` the bind address comes from ``config["server"]``;
    the stdio server needs no configuration at all.
    """
    from .search import run_search

    if transport == "streamable-http":
        server_cfg = config["server"]
        mcp = FastMCP(
            "chunksilo",
            host=server_cfg["host"],
            port=int(server_cfg["port"]),
            # Each POST is independent - no session affinity - so a tool call
            # never depends on which worker served the previous one.
            stateless_http=True,
            json_response=True,
        )
    else:
        mcp = FastMCP("chunksilo")

    @mcp.tool()
    async def search_docs(
        query: Annotated[str, Field(description="Search query text")],
        date_from: Annotated[str | None, Field(description="Optional start date filter (YYYY-MM-DD format, inclusive)")] = None,
        date_to: Annotated[str | None, Field(description="Optional end date filter (YYYY-MM-DD format, inclusive)")] = None,
    ) -> dict[str, Any]:
        """Search across all your indexed documentation using a natural language query."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: run_search(query, date_from, date_to, config_path=_server_config_path)
        )

    return mcp


def run_server(config_path: Path | None = None, transport_override: str | None = None):
    """Start the MCP server."""
    global _server_config_path, _mcp

    logger = logging.getLogger(__name__)

    from .cfgload import load_config

    # Load - and thereby activate - the configuration before importing
    # .search/.index: index.py reads it at import time via load_config().
    config = load_config(config_path)
    if config_path:
        _server_config_path = config_path

    # Logging goes to <storage_dir>/mcp.log, so it could not be configured
    # before the config was read.
    _setup_logging(config["storage"]["storage_dir"])
    logger.info("Starting ChunkSilo MCP server")
    transport = transport_override or config["server"]["transport"]

    if transport == "stdio":
        _mcp = _create_server(config, transport)
        _mcp.run()
        return

    if transport != "streamable-http":
        raise SystemExit(
            f"Unknown server.transport: {transport!r} (expected one of {', '.join(TRANSPORTS)})"
        )

    # Pay for the index, the models and the BM25 index once, before the first
    # request - concurrent first requests would otherwise all wait on the
    # slowest cold start.
    from . import search as search_module

    logger.info("Warming up the search pipeline before accepting requests")
    search_module.warm_up(config)

    _mcp = _create_server(config, transport)
    logger.info(
        "Serving MCP over streamable-http on %s:%s (no authentication - front it with a proxy before exposing it)",
        config["server"]["host"],
        config["server"]["port"],
    )
    _mcp.run(transport="streamable-http")


def main():
    """Entry point for the chunksilo-mcp command."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(
        prog="chunksilo-mcp",
        description="Run ChunkSilo MCP server (stdio or streamable-http transport)",
    )
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument(
        "--transport",
        choices=list(TRANSPORTS),
        help="Override server.transport from config",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    run_server(config_path, transport_override=args.transport)


if __name__ == "__main__":
    main()
