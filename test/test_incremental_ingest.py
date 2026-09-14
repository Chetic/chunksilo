import contextlib
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


# ===========================================================================
# Files must not be reprocessed on every run
#
# Three ways a build used to lose a file's state row and re-index it forever:
# it extracted no text, its directory was unavailable, or it could not be
# stat'd. All three showed up in production as "untouched files keep being
# re-indexed".
# ===========================================================================

class _LogSink(logging.Handler):
    """Collects records from chunksilo.index.

    Not a StreamHandler on purpose: IndexingUI mutes root StreamHandlers while
    it owns the terminal, which silences pytest's caplog for the whole build.
    """

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())

    @property
    def text(self) -> str:
        return "\n".join(self.messages)


@contextlib.contextmanager
def capture_index_log():
    sink = _LogSink()
    log = logging.getLogger("chunksilo.index")
    previous = log.level
    log.setLevel(logging.DEBUG)
    log.addHandler(sink)
    try:
        yield sink
    finally:
        log.removeHandler(sink)
        log.setLevel(previous)


def _tracked(db_path):
    """{path: (hash, doc_ids)} from the state DB."""
    with sqlite3.connect(db_path) as conn:
        return {
            row[0]: (row[1], row[2])
            for row in conn.execute("SELECT path, hash, doc_ids FROM files")
        }


def test_file_yielding_no_documents_is_recorded_once(test_env):
    """A file that extracts no text gets a state row and is not retried.

    Without this it is dropped from the state DB every run, looks new on the
    next one, and is reprocessed forever.
    """
    data_dir, _storage_dir, db_path = test_env
    create_file(data_dir, "empty.txt", "some text")
    real_load = index.LocalFileSystemSource.load_file

    def load_nothing(self, file_info, ctx=None):
        if file_info.path.endswith("empty.txt"):
            return []
        return real_load(self, file_info, ctx)

    with patch.object(index.LocalFileSystemSource, "load_file", load_nothing):
        build_index()
        row = _tracked(db_path).get(str((data_dir / "empty.txt").absolute()))
        assert row is not None, "empty file must still be tracked"
        assert row[1] == "", "an empty load stores no doc ids"

        with capture_index_log() as log:
            build_index()
        assert "Reprocessing" not in log.text
        assert "Indexing new file" not in log.text


def test_failed_file_is_retried(test_env):
    """A file that could not be read gets no state row, so it is retried."""
    data_dir, _storage_dir, db_path = test_env
    create_file(data_dir, "boom.txt", "some text")

    def load_boom(self, file_info, ctx=None):
        raise OSError("mount went away")

    with patch.object(index.LocalFileSystemSource, "load_file", load_boom):
        build_index()
    assert str((data_dir / "boom.txt").absolute()) not in _tracked(db_path)


def test_unavailable_directory_does_not_delete_its_files(test_env):
    """One unreachable directory must not wipe its files from the index.

    This is the destructive case: a laggy network mount used to prune the whole
    tree, which forced a full re-index on the following run. (When *every*
    directory is unavailable build_index bails out before touching anything;
    the damage needed a mixed config, which is what this sets up.)
    """
    data_dir, _storage_dir, db_path = test_env
    other_dir = data_dir.parent / "other"
    other_dir.mkdir()
    create_file(data_dir, "doc1.txt", "This is document 1.")
    create_file(other_dir, "doc2.txt", "This is document 2.")

    two_dirs = IndexConfig(
        directories=[DirectoryConfig(path=data_dir), DirectoryConfig(path=other_dir)],
        chunk_size=512,
        chunk_overlap=100,
    )
    real_is_available = index.LocalFileSystemSource.is_available

    def only_other_available(self):
        if self.base_dir == data_dir:
            self.scan_status.fail("availability check timed out after 30s")
            return False
        return real_is_available(self)

    with patch("chunksilo.index.load_index_config", return_value=two_dirs):
        build_index()
        assert len(_tracked(db_path)) == 2

        with patch.object(
            index.LocalFileSystemSource, "is_available", only_other_available
        ), capture_index_log() as log:
            build_index()

    assert len(_tracked(db_path)) == 2, "the mount was down, not the file"
    assert "did not see" in log.text


def test_stalled_walk_does_not_delete_its_files(test_env):
    """A directory walk that stalls halfway must not prune what it never reached."""
    data_dir, _storage_dir, db_path = test_env
    create_file(data_dir, "doc1.txt", "This is document 1.")
    create_file(data_dir, "doc2.txt", "This is document 2.")
    build_index()
    assert len(_tracked(db_path)) == 2

    def stalled_walk(self, on_event=None):
        # Yield nothing and report the scan as incomplete, as the real
        # _walk_with_timeout does when os.walk() stops responding.
        self.scan_status.fail("directory walk stalled after 30s")
        return iter(())

    with patch.object(index.LocalFileSystemSource, "_walk_with_timeout", stalled_walk):
        build_index()

    assert len(_tracked(db_path)) == 2


def test_unreadable_file_is_not_treated_as_deleted(test_env):
    """A file whose stat/hash fails is kept, not pruned and re-added next run."""
    data_dir, _storage_dir, db_path = test_env
    create_file(data_dir, "doc1.txt", "This is document 1.")
    create_file(data_dir, "doc2.txt", "This is document 2.")
    build_index()
    before = _tracked(db_path)
    assert len(before) == 2

    real_create = index.LocalFileSystemSource._create_file_info

    def flaky_create(self, file_path, tracked_files=None):
        if file_path.name == "doc2.txt":
            raise OSError("stale file handle")
        return real_create(self, file_path, tracked_files)

    with patch.object(index.LocalFileSystemSource, "_create_file_info", flaky_create):
        build_index()

    assert _tracked(db_path) == before


def test_genuinely_deleted_file_is_still_pruned(test_env):
    """Withholding deletions must not stop real deletions from being applied."""
    data_dir, _storage_dir, db_path = test_env
    create_file(data_dir, "doc1.txt", "This is document 1.")
    create_file(data_dir, "doc2.txt", "This is document 2.")
    build_index()
    assert len(_tracked(db_path)) == 2

    (data_dir / "doc2.txt").unlink()
    build_index()
    assert len(_tracked(db_path)) == 1


def test_reprocess_reason_is_logged(test_env):
    """Every reprocessed file says why, so a re-index loop is diagnosable."""
    data_dir, _storage_dir, _db_path = test_env
    create_file(data_dir, "doc1.txt", "This is document 1.")
    build_index()

    create_file(data_dir, "doc1.txt", "This is document 1, edited.")
    with capture_index_log() as log:
        build_index()
    assert "Reprocessing" in log.text
    assert "content changed" in log.text


def test_doc_conversion_failure_is_retried(test_env):
    """A .doc that could not be converted gets no state row, so it is tried
    again next run (for instance once LibreOffice is installed). Persisting it
    as a file that holds no text would silence it for ever."""
    data_dir, _storage_dir, db_path = test_env
    create_file(data_dir, "legacy.doc", "not really a doc")

    with patch("chunksilo.index._convert_doc_to_docx", return_value=None), \
         capture_index_log() as log:
        build_index()
    assert str((data_dir / "legacy.doc").absolute()) not in _tracked(db_path)
    assert "could not convert" in log.text

    with patch("chunksilo.index._convert_doc_to_docx", return_value=None) as convert:
        build_index()
    assert convert.called, "the failed .doc must be attempted again"


def test_timed_out_load_is_retried(test_env):
    """A file whose reader timed out gets no state row, so it is retried."""
    data_dir, _storage_dir, db_path = test_env
    create_file(data_dir, "slow.txt", "some text")
    real_run = index._run_with_timeout

    def timing_out(fn, timeout_seconds, default=index._SCAN_TIMEOUT_SENTINEL):
        if getattr(fn, "__name__", "") == "load_data":
            return default  # exactly what a timeout returns
        return real_run(fn, timeout_seconds, default)

    with patch.object(index, "_run_with_timeout", timing_out), capture_index_log() as log:
        build_index()
    assert str((data_dir / "slow.txt").absolute()) not in _tracked(db_path)
    assert "timed out" in log.text

    with capture_index_log() as log:
        build_index()
    assert "Indexing new file" in log.text
    assert str((data_dir / "slow.txt").absolute()) in _tracked(db_path)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can list a directory without read permission")
def test_unlistable_subdirectory_does_not_delete_its_files(test_env):
    """A subdirectory the walk cannot list must not have its files pruned.

    os.walk skips a directory it cannot open and carries on, so without an
    error callback the walk ends "cleanly" and everything indexed under that
    directory looks deleted - and is re-indexed once it is readable again.
    """
    data_dir, _storage_dir, db_path = test_env
    sub = data_dir / "sub"
    sub.mkdir()
    create_file(data_dir, "doc1.txt", "This is document 1.")
    create_file(sub, "doc2.txt", "This is document 2.")
    build_index()
    assert len(_tracked(db_path)) == 2

    sub.chmod(0)
    try:
        with capture_index_log() as log:
            build_index()
    finally:
        sub.chmod(0o755)
    assert len(_tracked(db_path)) == 2, "the directory was unlistable, not emptied"
    assert "did not see" in log.text

    with capture_index_log() as log:
        build_index()
    assert len(_tracked(db_path)) == 2
    assert "Indexing new file" not in log.text
