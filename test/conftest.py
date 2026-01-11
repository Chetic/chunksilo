"""Shared pytest fixtures for opd-mcp tests."""
import os
import pytest
from pathlib import Path
import sys

# Disable offline mode for tests that need to download models
# This must be set BEFORE any test file imports mcp_server
os.environ["OFFLINE"] = "0"

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def test_env(tmp_path):
    """Create isolated test environment with temp directories.

    Returns dict with:
        - data_dir: Path to temporary data directory
        - storage_dir: Path to temporary storage directory
        - db_path: Path to SQLite ingestion state database
    """
    data_dir = tmp_path / "data"
    storage_dir = tmp_path / "storage"
    data_dir.mkdir()
    storage_dir.mkdir()

    return {
        "data_dir": data_dir,
        "storage_dir": storage_dir,
        "db_path": storage_dir / "ingestion_state.db",
    }


@pytest.fixture
def mock_embed_model():
    """Create a Bag-of-Words mock embedding model for deterministic tests.

    This mock creates vectors based on word hashes, avoiding the need
    to load heavy ML models during tests.
    """
    from llama_index.core.embeddings import BaseEmbedding

    class BoWEmbedding(BaseEmbedding):
        """Simple Bag-of-Words embedding for testing."""

        def _get_vector(self, text: str):
            vec = [0.0] * 384
            for word in text.lower().split():
                model_idx = sum(ord(c) for c in word) % 384
                vec[model_idx] += 1.0
            return vec

        def _get_query_embedding(self, query: str):
            return self._get_vector(query)

        def _get_text_embedding(self, text: str):
            return self._get_vector(text)

        async def _aget_query_embedding(self, query: str):
            return self._get_vector(query)

    return BoWEmbedding(model_name="mock-bow")


@pytest.fixture
def mock_reranker():
    """Create a mock FlashRank reranker that passes through all results."""

    class MockReranker:
        """Pass-through reranker for testing."""

        def rerank(self, request):
            # Return all passages with score 1.0
            return [{"text": p["text"], "score": 1.0} for p in request.passages]

    return MockReranker()


@pytest.fixture
def sample_nodes():
    """Create sample NodeWithScore objects for testing retrieval logic."""
    from llama_index.core.schema import TextNode, NodeWithScore

    nodes = []
    for i in range(5):
        node = TextNode(
            text=f"Sample text content {i}. This is test content for node {i}.",
            metadata={
                "file_path": f"/path/to/file{i}.txt",
                "file_name": f"file{i}.txt",
                "creation_date": f"2024-01-{10+i:02d}",
                "last_modified_date": f"2024-06-{15+i:02d}",
            },
            id_=f"node_{i}"
        )
        nodes.append(NodeWithScore(node=node, score=0.9 - i * 0.1))
    return nodes


@pytest.fixture
def patched_ingest_globals(test_env):
    """Patch ingest module globals for isolated testing.

    This fixture patches DATA_DIR, STORAGE_DIR, and STATE_DB_PATH
    to use temporary directories, then restores originals after test.
    """
    import ingest

    original = {
        "DATA_DIR": ingest.DATA_DIR,
        "STORAGE_DIR": ingest.STORAGE_DIR,
        "STATE_DB_PATH": ingest.STATE_DB_PATH,
    }

    ingest.DATA_DIR = test_env["data_dir"]
    ingest.STORAGE_DIR = test_env["storage_dir"]
    ingest.STATE_DB_PATH = test_env["db_path"]

    yield test_env

    # Restore originals
    ingest.DATA_DIR = original["DATA_DIR"]
    ingest.STORAGE_DIR = original["STORAGE_DIR"]
    ingest.STATE_DB_PATH = original["STATE_DB_PATH"]


@pytest.fixture
def patched_mcp_globals(test_env):
    """Patch mcp_server module globals for isolated testing.

    This fixture patches STORAGE_DIR and resets caches to ensure
    tests don't interfere with each other.
    """
    import mcp_server

    original_storage = mcp_server.STORAGE_DIR
    original_initialized = mcp_server._embed_model_initialized

    mcp_server.STORAGE_DIR = test_env["storage_dir"]
    mcp_server._index_cache = None
    mcp_server._embed_model_initialized = False

    yield test_env

    mcp_server.STORAGE_DIR = original_storage
    mcp_server._index_cache = None
    mcp_server._embed_model_initialized = original_initialized
