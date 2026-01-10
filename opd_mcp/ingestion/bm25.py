"""BM25 index building for file name search."""

import logging
from pathlib import Path
from typing import Set

from llama_index.core.schema import TextNode

from opd_mcp.utils.text import tokenize_filename

logger = logging.getLogger(__name__)


def build_bm25_index(index, storage_dir: Path) -> None:
    """Build a BM25 index over file names from the docstore.

    This enables keyword matching for queries like 'cpp styleguide' to find
    files named 'cpp_styleguide.md'.

    Args:
        index: The LlamaIndex vector store index
        storage_dir: Directory to persist the BM25 index
    """
    from llama_index.retrievers.bm25 import BM25Retriever

    logger.info("Building BM25 index for file name matching...")

    # Create filename nodes - one per unique file
    filename_nodes = []
    seen_files: Set[str] = set()

    for doc_id, node in index.docstore.docs.items():
        metadata = node.metadata or {}
        file_name = metadata.get("file_name", "")
        file_path = metadata.get("file_path", "")

        if not file_name or file_path in seen_files:
            continue
        seen_files.add(file_path)

        tokens = tokenize_filename(file_name)
        filename_nodes.append(
            TextNode(
                text=" ".join(tokens),
                metadata={"file_name": file_name, "file_path": file_path},
                id_=f"bm25_{file_path}",
            )
        )

    if not filename_nodes:
        logger.warning("No documents found for BM25 indexing")
        return

    logger.info(f"Creating BM25 index with {len(filename_nodes)} file name entries")

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=filename_nodes,
        similarity_top_k=10,
    )

    bm25_dir = storage_dir / "bm25_index"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    bm25_retriever.persist(str(bm25_dir))

    logger.info(f"BM25 index persisted to {bm25_dir}")
