#!/usr/bin/env python3
"""
Ingestion pipeline for building a RAG index from PDF, DOCX, Markdown, and TXT documents.
Supports incremental indexing using a local SQLite database to track file states.
"""

import argparse
import logging
import sys
from typing import List, Set

# Check for --offline flag early and set environment variables BEFORE any imports
if "--offline" in sys.argv:
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter

# Import from refactored modules - use module references for testability
import opd_mcp.config as config
from opd_mcp.storage import IngestionState
from opd_mcp.storage.headings import get_heading_store
from opd_mcp.models.embeddings import (
    create_fastembed_embedding,
    ensure_embedding_model_cached,
)
from opd_mcp.models.reranking import ensure_rerank_model_cached
from opd_mcp.utils.progress import SimpleProgressBar, Spinner
from opd_mcp.ingestion import LocalFileSystemSource, build_bm25_index

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_index(download_only: bool = False, offline: bool = False) -> None:
    """Build and persist the vector index incrementally."""
    config.configure_offline_mode(offline, config.RETRIEVAL_MODEL_CACHE_DIR)

    logger.info(f"Starting ingestion from {config.DATA_DIR}")

    ensure_embedding_model_cached(config.RETRIEVAL_MODEL_CACHE_DIR, offline=offline)
    try:
        ensure_rerank_model_cached(config.RETRIEVAL_MODEL_CACHE_DIR, offline=offline)
    except FileNotFoundError:
        if download_only or offline:
            raise
        logger.warning("Rerank model could not be cached yet; continuing without it.")

    if download_only:
        logger.info("Models downloaded; skipping index build.")
        return

    # Initialize State and Data Source
    ingestion_state = IngestionState(config.STATE_DB_PATH)
    data_source = LocalFileSystemSource(config.DATA_DIR)

    # Initialize Embedding Model
    logger.info(f"Initializing embedding model: {config.RETRIEVAL_EMBED_MODEL_NAME}")
    with Spinner("Initializing embedding model"):
        embed_model = create_fastembed_embedding(config.RETRIEVAL_MODEL_CACHE_DIR, offline=offline)
    Settings.embed_model = embed_model

    # Configure Text Splitter
    text_splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separator=" ",
    )
    Settings.text_splitter = text_splitter

    # Load existing index or create new
    if (config.STORAGE_DIR / "docstore.json").exists():
        logger.info("Loading existing index context...")
        storage_context = StorageContext.from_defaults(persist_dir=str(config.STORAGE_DIR))
        index = load_index_from_storage(storage_context, embed_model=embed_model)
    else:
        logger.info("Creating new index context...")
        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex([], storage_context=storage_context, embed_model=embed_model)

    # Change Detection
    tracked_files = ingestion_state.get_all_files()
    found_files: Set[str] = set()
    files_to_process: List = []

    logger.info("Scanning for changes...")
    for file_info in data_source.iter_files():
        found_files.add(file_info.path)
        existing_state = tracked_files.get(file_info.path)

        if existing_state:
            # Check if modified
            if existing_state["hash"] != file_info.hash:
                logger.info(f"Modified file detected: {file_info.path}")
                files_to_process.append(file_info)
        else:
            # New file
            logger.info(f"New file detected: {file_info.path}")
            files_to_process.append(file_info)

    # Identify Deleted Files
    deleted_files = set(tracked_files.keys()) - found_files
    for deleted_path in deleted_files:
        logger.info(f"Deleted file detected: {deleted_path}")
        doc_ids = tracked_files[deleted_path]["doc_ids"]
        for doc_id in doc_ids:
            try:
                index.delete_ref_doc(doc_id, delete_from_docstore=True)
            except Exception as e:
                logger.warning(f"Failed to delete doc {doc_id} from index: {e}")
        # Clean up heading data for deleted file
        get_heading_store().remove_headings(deleted_path)
        ingestion_state.remove_file_state(deleted_path)

    if not files_to_process and not deleted_files:
        logger.info("No changes detected. Index is up to date.")
        return

    # Process New/Modified Files
    if files_to_process:
        progress = SimpleProgressBar(
            len(files_to_process), desc="Processing files", unit="file"
        )
        for file_info in files_to_process:
            # Remove old versions if they exist
            existing_state = tracked_files.get(file_info.path)
            if existing_state:
                for doc_id in existing_state["doc_ids"]:
                    try:
                        index.delete_ref_doc(doc_id, delete_from_docstore=True)
                    except KeyError:
                        pass  # Document might already be gone

            # Load and Index New Version
            docs = data_source.load_file(file_info)
            doc_ids = []
            for doc in docs:
                index.insert(doc)
                doc_ids.append(doc.doc_id)

            # Update State
            ingestion_state.update_file_state(file_info, doc_ids)
            progress.update()

    # Persist Index
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Persisting index to {config.STORAGE_DIR}")
    index.storage_context.persist(persist_dir=str(config.STORAGE_DIR))

    # Build BM25 index for file name matching
    build_bm25_index(index, config.STORAGE_DIR)

    logger.info("Ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the document index")
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download the retrieval models and exit",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run entirely offline",
    )
    args = parser.parse_args()

    try:
        build_index(download_only=args.download_models, offline=args.offline)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise
