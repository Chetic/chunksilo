"""BM25 retrieval for file name matching."""

import logging
from pathlib import Path
from typing import List, Optional

from llama_index.core.schema import NodeWithScore

from opd_mcp.config import BM25_INDEX_DIR

logger = logging.getLogger(__name__)

# Global cache for BM25 retriever
_bm25_retriever_cache = None


def ensure_bm25_retriever(index_dir: Path = None):
    """Load the BM25 retriever for file name matching.

    Args:
        index_dir: Path to BM25 index directory (defaults to BM25_INDEX_DIR)

    Returns:
        BM25Retriever instance, or None if index doesn't exist
    """
    global _bm25_retriever_cache

    if _bm25_retriever_cache is not None:
        return _bm25_retriever_cache

    index_dir = index_dir or BM25_INDEX_DIR

    if not index_dir.exists():
        logger.warning(f"BM25 index not found at {index_dir}. Run ingest.py to create it.")
        return None

    try:
        from llama_index.retrievers.bm25 import BM25Retriever

        logger.info(f"Loading BM25 index from {index_dir}")
        _bm25_retriever_cache = BM25Retriever.from_persist_dir(str(index_dir))
        logger.info("BM25 retriever loaded successfully")
        return _bm25_retriever_cache
    except Exception as e:
        logger.error(f"Failed to load BM25 retriever: {e}")
        return None


def expand_filename_matches(bm25_nodes: List[NodeWithScore], docstore) -> List[NodeWithScore]:
    """Expand BM25 file name matches to all chunks from those files.

    When BM25 matches a file name, we want to return all chunks from that file,
    not just the synthetic filename node.

    Args:
        bm25_nodes: BM25 search results (file name nodes)
        docstore: The document store containing all chunks

    Returns:
        List of NodeWithScore for all chunks from matched files
    """
    expanded = []
    matched_paths: set[str] = set()

    # Collect file paths from BM25 matches
    for bm25_node in bm25_nodes:
        file_path = bm25_node.node.metadata.get("file_path")
        if file_path:
            matched_paths.add(file_path)

    # Find all chunks from matched files
    for doc_id, node in docstore.docs.items():
        node_file_path = node.metadata.get("file_path")
        if node_file_path in matched_paths:
            # Give BM25-matched chunks a baseline score
            expanded.append(NodeWithScore(node=node, score=0.5))

    return expanded


def reset_bm25_retriever():
    """Reset the cached BM25 retriever (for testing)."""
    global _bm25_retriever_cache
    _bm25_retriever_cache = None
