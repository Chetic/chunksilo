#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
MCP server for querying documentation using RAG.
Returns raw document chunks for the calling LLM to synthesize.
"""
import argparse
import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field
from mcp.server.fastmcp import FastMCP

# Log file configuration
LOG_FILE = "mcp.log"
LOG_MAX_SIZE_MB = 10
LOG_MAX_SIZE_BYTES = LOG_MAX_SIZE_MB * 1024 * 1024

# Module-level state (set during initialization)
_server_config_path: Path | None = None
_mcp: FastMCP | None = None


def _rotate_log_if_needed():
    """Rotate log file if it exists and is over the size limit."""
    log_path = Path(LOG_FILE)
    if log_path.exists() and log_path.stat().st_size > LOG_MAX_SIZE_BYTES:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        process_id = os.getpid()
        rotated_name = f"mcp_{timestamp}_{process_id}.log"
        rotated_path = log_path.parent / rotated_name
        log_path.rename(rotated_path)
        log_path.touch()


def _setup_logging():
    """Configure logging for MCP server mode - file only, no stdout/stderr."""
    _rotate_log_if_needed()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
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

    if config_path:
        _server_config_path = config_path
        os.environ["CHUNKSILO_CONFIG"] = str(config_path)

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

    # Configure logging BEFORE importing anything that uses logging
    _setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Starting ChunkSilo MCP server")

    config_path = Path(args.config) if args.config else None
    run_server(config_path)


if __name__ == "__main__":
    main()
