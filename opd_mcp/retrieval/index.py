"""Index loading and management."""

import logging
from pathlib import Path
from typing import Optional

from llama_index.core import StorageContext, load_index_from_storage

import opd_mcp.config as config
from opd_mcp.models.embeddings import ensure_embed_model

logger = logging.getLogger(__name__)

# Global cache for the loaded index
_index_cache = None


def load_llamaindex_index(storage_dir: Path = None):
    """Load the LlamaIndex from storage.

    Args:
        storage_dir: Path to storage directory (defaults to config.STORAGE_DIR)

    Returns:
        The loaded VectorStoreIndex

    Raises:
        FileNotFoundError: If storage directory doesn't exist
    """
    global _index_cache

    if _index_cache is not None:
        return _index_cache

    storage_dir = storage_dir or config.STORAGE_DIR

    if not storage_dir.exists():
        raise FileNotFoundError(
            f"Storage directory {storage_dir} does not exist. "
            "Please run ingest.py first."
        )

    logger.info("Loading LlamaIndex from storage...")

    # Make sure the embedding model is configured before using the index
    ensure_embed_model()

    storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
    index = load_index_from_storage(storage_context)
    _index_cache = index
    return index


def reset_index_cache():
    """Reset the cached index (for testing)."""
    global _index_cache
    _index_cache = None


def get_index_cache():
    """Get the current cached index (may be None)."""
    return _index_cache
