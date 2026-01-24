#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
MCP server for querying documentation using RAG.
Returns raw document chunks for the calling LLM to synthesize.
"""
import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field
from mcp.server.fastmcp import FastMCP

from .search import run_search

# Log file configuration
LOG_FILE = "mcp.log"
LOG_MAX_SIZE_MB = 10
LOG_MAX_SIZE_BYTES = LOG_MAX_SIZE_MB * 1024 * 1024


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


# Rotate log file if needed before setting up logging
_rotate_log_if_needed()

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
logging.getLogger("llama_index.readers.confluence").setLevel(logging.WARNING)

# Initialize FastMCP server
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
        None, run_search, query, date_from, date_to
    )


def run_server(config_path: Path | None = None):
    """Start the MCP server."""
    if config_path:
        os.environ["CHUNKSILO_CONFIG"] = str(config_path)
    mcp.run()


if __name__ == "__main__":
    run_server()
