import os
import shutil
import sqlite3
import logging
import sys
from pathlib import Path
import pytest

# Add parent directory to path to import ingest modules
sys.path.append(str(Path(__file__).parent.parent))

import ingest
from ingest import build_index, IngestionState

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_incremental")

@pytest.fixture
def test_env(tmp_path):
    """Setup test environment with temporary data and storage directories."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    storage_dir = tmp_path / "storage"
    db_path = storage_dir / "ingestion_state.db"
    
    # Save original globals
    orig_data_dir = ingest.DATA_DIR
    orig_storage_dir = ingest.STORAGE_DIR
    orig_db_path = ingest.STATE_DB_PATH
    orig_cache_dir = ingest.RETRIEVAL_MODEL_CACHE_DIR
    
    # Set globals to test paths
    ingest.DATA_DIR = data_dir
    ingest.STORAGE_DIR = storage_dir
    ingest.STATE_DB_PATH = db_path
    
    # Point cache to real models directory to avoid re-downloading
    project_root = Path(__file__).parent.parent
    ingest.RETRIEVAL_MODEL_CACHE_DIR = project_root / "models"
    
    yield data_dir, storage_dir, db_path
    
    # Restore globals
    ingest.DATA_DIR = orig_data_dir
    ingest.STORAGE_DIR = orig_storage_dir
    ingest.STATE_DB_PATH = orig_db_path
    ingest.RETRIEVAL_MODEL_CACHE_DIR = orig_cache_dir

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
    build_index(offline=True)
    check_db_count(db_path, 1)

    logger.info("--- Step 2: No Change Run ---")
    build_index(offline=True)
    check_db_count(db_path, 1)

    logger.info("--- Step 3: Add File ---")
    create_file(data_dir, "doc2.txt", "This is document 2.")
    build_index(offline=True)
    check_db_count(db_path, 2)

    logger.info("--- Step 4: Modify File ---")
    create_file(data_dir, "doc1.txt", "This is document 1 modified.")
    build_index(offline=True)
    check_db_count(db_path, 2)
    
    # Check if hash changed in DB
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT hash FROM files WHERE path LIKE '%doc1.txt'").fetchone()
        assert row is not None
        logger.info(f"Doc1 Hash: {row[0]}")

    logger.info("--- Step 5: Delete File ---")
    (data_dir / "doc2.txt").unlink()
    build_index(offline=True)
    check_db_count(db_path, 1)
