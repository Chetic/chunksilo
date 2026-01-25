import os
import shutil
import sqlite3
import logging
from pathlib import Path
from unittest.mock import patch
import pytest

from chunksilo import index
from chunksilo.index import build_index, IngestionState, IndexConfig, DirectoryConfig

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_incremental")


def _create_mock_embedding():
    """Create a lightweight mock embedding model for testing."""
    from llama_index.core.embeddings import BaseEmbedding

    class MockEmbedding(BaseEmbedding):
        """Simple mock embedding that creates deterministic vectors from text."""

        def _get_vector(self, text: str):
            vec = [0.0] * 384
            for word in text.lower().split():
                idx = sum(ord(c) for c in word) % 384
                vec[idx] += 1.0
            return vec

        def _get_query_embedding(self, query: str):
            return self._get_vector(query)

        def _get_text_embedding(self, text: str):
            return self._get_vector(text)

        async def _aget_query_embedding(self, query: str):
            return self._get_vector(query)

    return MockEmbedding(model_name="mock-test")


@pytest.fixture
def test_env(tmp_path):
    """Setup test environment with temporary data and storage directories.

    Uses mock embedding model to avoid needing real models in cache.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    storage_dir = tmp_path / "storage"
    db_path = storage_dir / "ingestion_state.db"

    # Create test IndexConfig pointing to test data dir
    test_index_config = IndexConfig(
        directories=[DirectoryConfig(path=data_dir)],
        chunk_size=512,
        chunk_overlap=100
    )

    # Save original globals
    orig_storage_dir = index.STORAGE_DIR
    orig_db_path = index.STATE_DB_PATH

    # Set globals to test paths
    index.STORAGE_DIR = storage_dir
    index.STATE_DB_PATH = db_path

    # Create mock embedding model
    mock_embed = _create_mock_embedding()

    # Patch load_index_config to return test config, and mock embedding functions
    with patch("chunksilo.index.load_index_config", return_value=test_index_config), \
         patch("chunksilo.index._create_fastembed_embedding", return_value=mock_embed), \
         patch("chunksilo.index.ensure_embedding_model_cached"), \
         patch("chunksilo.index.ensure_rerank_model_cached"):
        yield data_dir, storage_dir, db_path

    # Restore globals
    index.STORAGE_DIR = orig_storage_dir
    index.STATE_DB_PATH = orig_db_path

def create_file(data_dir, name, content):
    path = data_dir / name
    path.write_text(content)
    return path

def check_db_count(db_path, expected_count):
    if not db_path.exists():
         if expected_count == 0: return
         raise AssertionError(f"DB not found but expected {expected_count} files")
    
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT count(*) FROM files").fetchone()[0]
        assert count == expected_count, f"Expected {expected_count} files in DB, found {count}"

def test_incremental_ingestion(test_env):
    data_dir, storage_dir, db_path = test_env
    
    logger.info("--- Step 1: Initial Run (1 file) ---")
    create_file(data_dir, "doc1.txt", "This is document 1.")
    build_index()
    check_db_count(db_path, 1)

    logger.info("--- Step 2: No Change Run ---")
    build_index()
    check_db_count(db_path, 1)

    logger.info("--- Step 3: Add File ---")
    create_file(data_dir, "doc2.txt", "This is document 2.")
    build_index()
    check_db_count(db_path, 2)

    logger.info("--- Step 4: Modify File ---")
    create_file(data_dir, "doc1.txt", "This is document 1 modified.")
    build_index()
    check_db_count(db_path, 2)
    
    # Check if hash changed in DB
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT hash FROM files WHERE path LIKE '%doc1.txt'").fetchone()
        assert row is not None
        logger.info(f"Doc1 Hash: {row[0]}")

    logger.info("--- Step 5: Delete File ---")
    (data_dir / "doc2.txt").unlink()
    build_index()
    check_db_count(db_path, 1)
