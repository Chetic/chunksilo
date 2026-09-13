#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
MCP server for querying documentation using RAG.
Returns raw document chunks for the calling LLM to synthesize.
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


def _create_server() -> FastMCP:
    """Create and configure the MCP server with tools."""
    from .search import run_search

    mcp = FastMCP("llamaindex-docs-rag")

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


def run_server(config_path: Path | None = None):
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

    _mcp = _create_server()
    _mcp.run()


def main():
    """Entry point for the chunksilo-mcp command."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(
        prog="chunksilo-mcp",
        description="Run ChunkSilo MCP server (stdio transport)",
    )
    parser.add_argument("--config", help="Path to config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    run_server(config_path)


if __name__ == "__main__":
    main()
