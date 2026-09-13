#!/usr/bin/env python3
"""Integration tests for revision-aware indexing.

Superseded revisions and review copies are filtered out of the scan, so the
existing deletion loop prunes previously-indexed ones, and an older revision
comes back automatically when the newest file disappears.
"""

import contextlib
import copy
import json
import logging
import os
import sqlite3
from unittest.mock import patch

import pytest

from chunksilo import index
from chunksilo.index import build_index, IndexConfig, DirectoryConfig


def _create_mock_embedding():
    from llama_index.core.embeddings import BaseEmbedding

    class MockEmbedding(BaseEmbedding):
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


class _LogSink(logging.Handler):
    """Collects records from chunksilo.index (see test_incremental_ingest)."""

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


@contextlib.contextmanager
def versioning_config(**overrides):
    """Run with indexing.versioning options overridden in index._config."""
    orig = index._config
    cfg = copy.deepcopy(orig)
    cfg.setdefault("indexing", {}).setdefault("versioning", {}).update(overrides)
    index._config = cfg
    try:
        yield
    finally:
        index._config = orig


@pytest.fixture
def test_env(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    storage_dir = tmp_path / "storage"
    db_path = storage_dir / "ingestion_state.db"

    test_index_config = IndexConfig(
        directories=[DirectoryConfig(path=data_dir)],
        chunk_size=512,
        chunk_overlap=100,
    )

    orig_storage_dir = index.STORAGE_DIR
    orig_db_path = index.STATE_DB_PATH

    index.STORAGE_DIR = storage_dir
    index.STATE_DB_PATH = db_path

    mock_embed = _create_mock_embedding()

    with patch("chunksilo.index.load_index_config", return_value=test_index_config), \
         patch("chunksilo.index._create_fastembed_embedding", return_value=mock_embed), \
         patch("chunksilo.index.ensure_embedding_model_cached"), \
         patch("chunksilo.index.ensure_rerank_model_cached"):
        yield data_dir, storage_dir, db_path

    index.STORAGE_DIR = orig_storage_dir
    index.STATE_DB_PATH = orig_db_path


def create_file(data_dir, name, content, mtime=None):
    path = data_dir / name
    path.write_text(content)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def tracked_names(db_path):
    if not db_path.exists():
        return set()
    with sqlite3.connect(db_path) as conn:
        return {
            os.path.basename(row[0])
            for row in conn.execute("SELECT path FROM files")
        }


def docstore_metadata(storage_dir):
    """{file_name: merged metadata} straight from the persisted docstore."""
    docstore = json.loads((storage_dir / "docstore.json").read_text())
    result = {}
    for entry in docstore.get("docstore/data", {}).values():
        metadata = entry.get("__data__", {}).get("metadata", {})
        name = metadata.get("file_name")
        if name:
            result.setdefault(name, {}).update(metadata)
    return result


T0, T1, T2, T3 = 1_600_000_000.0, 1_600_100_000.0, 1_600_200_000.0, 1_600_300_000.0


def test_fresh_build_keeps_only_newest(test_env):
    data_dir, storage_dir, db_path = test_env
    create_file(data_dir, "Spec PB1.txt", "spec preliminary one", mtime=T0)
    create_file(data_dir, "Spec PB2.txt", "spec preliminary two", mtime=T1)
    create_file(data_dir, "Spec B.txt", "spec released", mtime=T2)
    create_file(data_dir, "Spec B (review comments JD).txt", "commented", mtime=T3)
    create_file(data_dir, "Other.txt", "unrelated document", mtime=T0)

    build_index()

    assert tracked_names(db_path) == {"Spec B.txt", "Other.txt"}
    metadata = docstore_metadata(storage_dir)
    assert set(metadata) == {"Spec B.txt", "Other.txt"}
    assert metadata["Spec B.txt"]["doc_group"].startswith("path:")
    assert "superseded" not in metadata["Spec B.txt"]


def test_revision_dirs_group_and_newest_wins(test_env):
    data_dir, storage_dir, db_path = test_env
    (data_dir / "Spec" / "PB1").mkdir(parents=True)
    (data_dir / "Spec" / "B").mkdir(parents=True)
    create_file(data_dir, "Spec/PB1/Spec.txt", "preliminary", mtime=T0)
    create_file(data_dir, "Spec/B/Spec.txt", "released", mtime=T1)

    build_index()

    with sqlite3.connect(db_path) as conn:
        paths = [row[0] for row in conn.execute("SELECT path FROM files")]
    assert len(paths) == 1 and paths[0].endswith("B/Spec.txt")


def test_new_revision_prunes_old(test_env):
    data_dir, _storage_dir, db_path = test_env
    create_file(data_dir, "Spec PB1.txt", "preliminary", mtime=T0)
    build_index()
    assert tracked_names(db_path) == {"Spec PB1.txt"}

    create_file(data_dir, "Spec B.txt", "released", mtime=T1)
    build_index()
    assert tracked_names(db_path) == {"Spec B.txt"}


def test_newest_deleted_resurrects_older(test_env):
    data_dir, _storage_dir, db_path = test_env
    create_file(data_dir, "Spec PB1.txt", "preliminary", mtime=T0)
    create_file(data_dir, "Spec B.txt", "released", mtime=T1)
    build_index()
    assert tracked_names(db_path) == {"Spec B.txt"}

    (data_dir / "Spec B.txt").unlink()
    build_index()
    assert tracked_names(db_path) == {"Spec PB1.txt"}


def test_first_versioned_run_prunes_existing(test_env):
    """Upgrade path: files indexed before versioning existed get pruned."""
    data_dir, _storage_dir, db_path = test_env
    create_file(data_dir, "Spec PB1.txt", "preliminary", mtime=T0)
    create_file(data_dir, "Spec B.txt", "released", mtime=T1)
    with versioning_config(enabled=False):
        build_index()
    assert tracked_names(db_path) == {"Spec PB1.txt", "Spec B.txt"}

    build_index()
    assert tracked_names(db_path) == {"Spec B.txt"}


def test_disabled_indexes_everything(test_env):
    data_dir, _storage_dir, db_path = test_env
    create_file(data_dir, "Spec PB1.txt", "preliminary", mtime=T0)
    create_file(data_dir, "Spec B.txt", "released", mtime=T1)
    create_file(data_dir, "Copy of Spec.txt", "copied", mtime=T2)
    with versioning_config(enabled=False):
        build_index()
    assert tracked_names(db_path) == {"Spec PB1.txt", "Spec B.txt", "Copy of Spec.txt"}


def test_index_superseded_stamps_and_tracks_flips(test_env):
    data_dir, storage_dir, db_path = test_env
    create_file(data_dir, "Spec PB1.txt", "preliminary", mtime=T0)
    create_file(data_dir, "Spec B.txt", "released", mtime=T1)
    with versioning_config(index_superseded=True):
        build_index()
    assert tracked_names(db_path) == {"Spec PB1.txt", "Spec B.txt"}
    metadata = docstore_metadata(storage_dir)
    assert metadata["Spec PB1.txt"]["superseded"] == "true"
    assert "superseded" not in metadata["Spec B.txt"]
    assert metadata["Spec PB1.txt"]["doc_group"] == metadata["Spec B.txt"]["doc_group"]

    # A newer revision arrives: the previous latest is unchanged on disk but
    # its stamp must flip, so it is reprocessed.
    create_file(data_dir, "Spec C.txt", "newer release", mtime=T2)
    with versioning_config(index_superseded=True):
        with capture_index_log() as log:
            build_index()
    assert "revision status changed" in log.text
    metadata = docstore_metadata(storage_dir)
    assert metadata["Spec B.txt"]["superseded"] == "true"
    assert "superseded" not in metadata["Spec C.txt"]

    # Turning the option back off prunes the old revisions.
    build_index()
    assert tracked_names(db_path) == {"Spec C.txt"}


def test_stale_stamp_cleaned_when_versioning_disabled(test_env):
    """Disabling versioning entirely also clears leftover superseded stamps."""
    data_dir, storage_dir, db_path = test_env
    create_file(data_dir, "Spec PB1.txt", "preliminary", mtime=T0)
    create_file(data_dir, "Spec B.txt", "released", mtime=T1)
    with versioning_config(index_superseded=True):
        build_index()
    assert docstore_metadata(storage_dir)["Spec PB1.txt"]["superseded"] == "true"

    with versioning_config(enabled=False):
        with capture_index_log() as log:
            build_index()
    assert "revision status changed" in log.text
    assert "superseded" not in docstore_metadata(storage_dir)["Spec PB1.txt"]


def test_unchanged_keeper_not_reprocessed(test_env):
    data_dir, _storage_dir, _db_path = test_env
    create_file(data_dir, "Spec PB1.txt", "preliminary", mtime=T0)
    create_file(data_dir, "Spec B.txt", "released", mtime=T1)
    build_index()

    with capture_index_log() as log:
        build_index()
    assert "Reprocessing" not in log.text
    assert "Indexing new file" not in log.text


def test_skip_reasons_logged(test_env):
    data_dir, _storage_dir, _db_path = test_env
    create_file(data_dir, "Spec PB1.txt", "preliminary", mtime=T0)
    create_file(data_dir, "Spec B.txt", "released", mtime=T1)
    create_file(data_dir, "Copy of Spec.txt", "copied", mtime=T2)

    with capture_index_log() as log:
        build_index()
    assert "Skipping superseded revision" in log.text
    assert "Skipping review copy" in log.text
