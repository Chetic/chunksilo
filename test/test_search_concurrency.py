#!/usr/bin/env python3
"""Concurrent first-use of the search pipeline must not duplicate cold starts.

An HTTP server dispatches every tool call into a thread pool with no session
affinity, so several clients can arrive before anything is loaded. Each loader
used to be an unguarded module global, so N simultaneous first calls each built
their own index, ONNX session and reranker - N times the work and memory, with
N-1 of each discarded. Three parallel calls that way exceed a 60-second client
timeout on a cold server.
"""
import threading
import time

import pytest

from chunksilo import search


@pytest.fixture(autouse=True)
def clean_caches(monkeypatch):
    """Start every test with an unloaded pipeline and restore afterwards."""
    monkeypatch.setattr(search, "_index_cache", None)
    monkeypatch.setattr(search, "_embed_model_initialized", False)
    monkeypatch.setattr(search, "_reranker_model", None)
    monkeypatch.setattr(search, "_bm25_retriever_cache", None)
    yield


def _run_concurrently(fn, threads=6):
    """Call fn from `threads` threads released at the same moment."""
    start = threading.Barrier(threads)
    errors = []

    def target():
        try:
            start.wait(timeout=10)
            fn()
        except Exception as exc:  # pragma: no cover - surfaced by the assert
            errors.append(exc)

    workers = [threading.Thread(target=target) for _ in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=30)
    assert not errors, errors


class _Counter:
    """A constructor stand-in that is slow enough to overlap callers."""

    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, *_args, **_kwargs):
        with self._lock:
            self.calls += 1
        time.sleep(0.05)
        return object()


class _Settings:
    """Stand-in for llama_index Settings, which type-checks embed_model."""

    embed_model = None


def test_embed_model_loads_once(monkeypatch):
    counter = _Counter()
    monkeypatch.setattr(search, "FastEmbedEmbedding", counter)
    monkeypatch.setattr(search, "_get_cached_model_path", lambda *_a, **_k: None)
    monkeypatch.setattr(search, "Settings", _Settings)
    config = {
        "retrieval": {"embed_model_name": "m", "offline": False},
        "storage": {"model_cache_dir": "/nonexistent"},
    }

    _run_concurrently(lambda: search._ensure_embed_model(config))

    assert counter.calls == 1
    assert search._embed_model_initialized is True


def test_reranker_loads_once(monkeypatch):
    counter = _Counter()
    monkeypatch.setattr("flashrank.Ranker", counter)
    config = {
        "retrieval": {"rerank_model_name": "ms-marco-MiniLM-L-12-v2", "offline": False},
        "storage": {"model_cache_dir": "/nonexistent"},
    }

    _run_concurrently(lambda: search._ensure_reranker(config))

    assert counter.calls == 1


def test_index_loads_once(monkeypatch, tmp_path):
    counter = _Counter()
    monkeypatch.setattr(search, "load_index_from_storage", counter)
    monkeypatch.setattr(search.StorageContext, "from_defaults", lambda **_k: object())
    monkeypatch.setattr(search, "_ensure_embed_model", lambda _c: None)
    config = {"storage": {"storage_dir": str(tmp_path)}}

    _run_concurrently(lambda: search.load_llamaindex_index(config))

    assert counter.calls == 1


def test_failed_reranker_load_is_not_cached(monkeypatch):
    """A loader that raises must leave the cache empty so the next call retries.

    Double-checked locking is easy to get wrong here: assigning the global
    before the constructor returns would publish a half-built object.
    """
    attempts = []

    def boom(*_args, **_kwargs):
        attempts.append(1)
        raise RuntimeError("no model in cache")

    monkeypatch.setattr("flashrank.Ranker", boom)
    config = {
        "retrieval": {"rerank_model_name": "ms-marco-MiniLM-L-12-v2", "offline": False},
        "storage": {"model_cache_dir": "/nonexistent"},
    }

    for _ in range(2):
        with pytest.raises(RuntimeError):
            search._ensure_reranker(config)

    assert len(attempts) == 2
    assert search._reranker_model is None


def test_warm_up_loads_everything(monkeypatch, tmp_path):
    """warm_up must populate every cache a query needs."""
    loaded = []
    monkeypatch.setattr(search, "load_llamaindex_index", lambda _c: loaded.append("index"))
    monkeypatch.setattr(search, "_ensure_embed_model", lambda _c: loaded.append("embed"))
    monkeypatch.setattr(search, "_ensure_reranker", lambda _c: loaded.append("rerank"))
    monkeypatch.setattr(search, "_ensure_bm25_retriever", lambda _c: loaded.append("bm25"))
    monkeypatch.setattr(search, "get_heading_store", lambda: loaded.append("headings"))
    config = {
        "retrieval": {"offline": False},
        "storage": {"model_cache_dir": str(tmp_path), "storage_dir": str(tmp_path)},
    }

    search.warm_up(config)

    assert loaded == ["index", "embed", "rerank", "bm25", "headings"]


def test_warm_up_survives_a_failure(monkeypatch, tmp_path):
    """A server that cannot preload is still worth starting."""
    def boom(*_a, **_k):
        raise RuntimeError("storage missing")

    monkeypatch.setattr(search, "load_llamaindex_index", boom)
    monkeypatch.setattr(search, "_ensure_embed_model", lambda _c: None)
    monkeypatch.setattr(search, "_ensure_reranker", lambda _c: None)
    monkeypatch.setattr(search, "_ensure_bm25_retriever", lambda _c: None)
    monkeypatch.setattr(search, "get_heading_store", lambda: None)
    config = {
        "retrieval": {"offline": False},
        "storage": {"model_cache_dir": str(tmp_path), "storage_dir": str(tmp_path)},
    }

    search.warm_up(config)  # must not raise
