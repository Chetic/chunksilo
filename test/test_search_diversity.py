#!/usr/bin/env python3
"""Tests for per-document result diversification and superseded demotion.

One document's many chunks (or its many revisions) must not fill the whole
top-k; superseded revisions rank behind current documents.
"""

import copy
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore, TextNode

from chunksilo import index, search
from chunksilo.index import build_index, DirectoryConfig, IndexConfig
from chunksilo.search import (
    _diversify_by_group,
    _format_bm25_matches,
    _partition_superseded,
    run_search,
)


def _node(text, score=1.0, **metadata):
    return NodeWithScore(node=TextNode(text=text, metadata=metadata), score=score)


# =============================================================================
# Unit tests for the helpers
# =============================================================================


class TestDiversifyByGroup:
    def test_cap_enforced_and_next_group_pulled_in(self):
        nodes = [
            _node("a1", doc_group="A"),
            _node("a2", doc_group="A"),
            _node("a3", doc_group="A"),
            _node("b1", doc_group="B"),
        ]
        result = _diversify_by_group(nodes, top_k=3, max_per_doc=2)
        assert [n.node.get_content() for n in result] == ["a1", "a2", "b1"]

    def test_backfill_keeps_top_k_full(self):
        nodes = [_node(f"a{i}", doc_group="A") for i in range(6)]
        result = _diversify_by_group(nodes, top_k=4, max_per_doc=2)
        assert [n.node.get_content() for n in result] == ["a0", "a1", "a2", "a3"]

    def test_backfill_preserves_rank_order(self):
        nodes = [
            _node("a1", doc_group="A"),
            _node("a2", doc_group="A"),
            _node("a3", doc_group="A"),
            _node("b1", doc_group="B"),
        ]
        result = _diversify_by_group(nodes, top_k=4, max_per_doc=2)
        assert [n.node.get_content() for n in result] == ["a1", "a2", "b1", "a3"]

    def test_file_path_fallback_for_unstamped_chunks(self):
        nodes = [
            _node("x1", file_path="/s/x.txt"),
            _node("x2", file_path="/s/x.txt"),
            _node("y1", file_path="/s/y.txt"),
        ]
        result = _diversify_by_group(nodes, top_k=2, max_per_doc=1)
        assert [n.node.get_content() for n in result] == ["x1", "y1"]

    def test_zero_disables_diversity_but_still_truncates(self):
        nodes = [_node(f"a{i}", doc_group="A") for i in range(6)]
        result = _diversify_by_group(nodes, top_k=5, max_per_doc=0)
        assert len(result) == 5
        assert [n.node.get_content() for n in result] == ["a0", "a1", "a2", "a3", "a4"]


class TestPartitionSuperseded:
    def test_stable_demotion(self):
        nodes = [
            _node("old1", superseded="true"),
            _node("fresh1"),
            _node("old2", superseded="true"),
            _node("fresh2"),
        ]
        result = _partition_superseded(nodes)
        assert [n.node.get_content() for n in result] == [
            "fresh1", "fresh2", "old1", "old2",
        ]

    def test_no_op_without_superseded(self):
        nodes = [_node("a"), _node("b")]
        assert _partition_superseded(nodes) is nodes


class TestBm25MatchedFiles:
    def _config(self):
        from chunksilo.cfgload import _DEFAULTS, _deep_merge

        return _deep_merge(_DEFAULTS, {})

    def test_dedupe_by_group_and_drop_superseded(self):
        nodes = [
            _node("m", 2.0, file_path="/s/manual_v2.txt", doc_group="g1"),
            _node("m", 1.5, file_path="/s/manual_v1.txt", doc_group="g1"),
            _node("m", 1.2, file_path="/s/old_A.txt", doc_group="g2", superseded="true"),
            _node("m", 1.0, file_path="/s/other.txt", doc_group="g3"),
        ]
        matches = _format_bm25_matches(nodes, self._config())
        names = [Path(m["uri"]).name for m in matches]
        assert names == ["manual_v2.txt", "other.txt"]

    def test_ungrouped_files_kept(self):
        nodes = [
            _node("m", 2.0, file_path="/s/a.txt"),
            _node("m", 1.0, file_path="/s/b.txt"),
        ]
        matches = _format_bm25_matches(nodes, self._config())
        assert len(matches) == 2


# =============================================================================
# End-to-end through run_search on a real mini index
# =============================================================================


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


class FakeRanker:
    """Equal scores: rerank order collapses to retrieval order."""

    def rerank(self, request):
        return [{"id": i, "score": 0.9} for i in range(len(request.passages))]


T0, T1 = 1_600_000_000.0, 1_600_100_000.0


def _search_config(storage_dir, model_dir):
    from chunksilo.cfgload import _DEFAULTS, _deep_merge

    return _deep_merge(
        _DEFAULTS,
        {
            "storage": {"storage_dir": str(storage_dir), "model_cache_dir": str(model_dir)},
            # embed_top_k above the corpus size so every chunk (including the
            # small overview doc) reaches the diversifier deterministically.
            "retrieval": {"recency_boost": 0.0, "embed_top_k": 100},
        },
    )


def _build_corpus(base: Path) -> Path:
    """Two revisions of a many-chunk manual plus one small overview doc."""
    data_dir = base / "data"
    storage_dir = base / "storage"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ~7 chunks per revision at chunk_size 64: enough for the manual to crowd
    # out the overview without a cap, while all chunks (2 revisions + the
    # overview) still fit within embed_top_k so the diversifier sees them.
    manual = " ".join(
        f"widget assembly manual step {i}: attach the widget bracket and torque "
        f"the widget bolts to spec."
        for i in range(25)
    )
    for name, content, mtime in [
        ("widget_manual_v1.txt", manual + " preliminary draft.", T0),
        ("widget_manual_v2.txt", manual + " released edition.", T1),
        ("widget_overview.txt", "widget overview: a short description of the widget product line.", T0),
    ]:
        path = data_dir / name
        path.write_text(content)
        os.utime(path, (mtime, mtime))

    test_index_config = IndexConfig(
        directories=[DirectoryConfig(path=data_dir)], chunk_size=64, chunk_overlap=8
    )
    orig = (
        index.STORAGE_DIR,
        index.STATE_DB_PATH,
        index.BM25_INDEX_DIR,
        index.HEADING_STORE_PATH,
        index._config,
    )
    index.STORAGE_DIR = storage_dir
    index.STATE_DB_PATH = storage_dir / "ingestion_state.db"
    index.BM25_INDEX_DIR = storage_dir / "bm25_index"
    index.HEADING_STORE_PATH = storage_dir / "heading_store.json"
    index._heading_store = None
    cfg = copy.deepcopy(index._config)
    # Keep both revisions in the index so demotion is observable.
    cfg.setdefault("indexing", {}).setdefault("versioning", {})["index_superseded"] = True
    index._config = cfg

    mock_embed = _create_mock_embedding()
    try:
        with patch("chunksilo.index.load_index_config", return_value=test_index_config), \
             patch("chunksilo.index._create_fastembed_embedding", return_value=mock_embed), \
             patch("chunksilo.index.ensure_embedding_model_cached"), \
             patch("chunksilo.index.ensure_rerank_model_cached"):
            build_index()
    finally:
        (
            index.STORAGE_DIR,
            index.STATE_DB_PATH,
            index.BM25_INDEX_DIR,
            index.HEADING_STORE_PATH,
            index._config,
        ) = orig
        index._heading_store = None
    return storage_dir


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    base = tmp_path_factory.mktemp("diversity_corpus")
    storage_dir = _build_corpus(base)
    return base, storage_dir


@pytest.fixture
def search_env(corpus, monkeypatch):
    base, storage_dir = corpus
    config = _search_config(storage_dir, base / "models")

    monkeypatch.setattr(search, "_config", config)
    monkeypatch.setattr(search, "_index_cache", None)
    monkeypatch.setattr(search, "_bm25_retriever_cache", None)
    monkeypatch.setattr(search, "_embed_model_initialized", True)
    monkeypatch.setattr(search, "_reranker_model", FakeRanker())
    monkeypatch.setattr(index, "HEADING_STORE_PATH", storage_dir / "heading_store.json")
    monkeypatch.setattr(index, "_heading_store", None)
    Settings.embed_model = _create_mock_embedding()
    yield config


def result_names(result) -> list[str]:
    assert "error" not in result, result.get("error")
    return [Path(c["location"]["uri"] or "").name for c in result["chunks"]]


QUERY = "widget assembly manual"


class TestSearchDiversity:
    def test_results_vary_across_documents(self, search_env):
        names = result_names(run_search(QUERY))
        # The manual revisions share one doc_group: at most 2 of its chunks,
        # and the small overview document still makes the cut.
        assert "widget_overview.txt" in names
        assert 0 < len(names) <= search_env["retrieval"]["rerank_top_k"]

    def test_superseded_revision_never_shown_over_current(self, search_env):
        names = result_names(run_search(QUERY))
        assert "widget_manual_v1.txt" not in names
        assert "widget_manual_v2.txt" in names

    def test_cap_disabled_restores_flat_ranking(self, search_env, monkeypatch):
        config = copy.deepcopy(search_env)
        config["retrieval"]["max_chunks_per_doc"] = 0
        monkeypatch.setattr(search, "_config", config)
        names = result_names(run_search(QUERY))
        # Without the cap the many manual chunks crowd out the overview.
        assert "widget_overview.txt" not in names
        assert len(names) == config["retrieval"]["rerank_top_k"]

    def test_bm25_matched_files_deduped_per_group(self, search_env):
        result = run_search("widget manual")
        matched = [Path(f["uri"]).name for f in result["matched_files"]]
        # One entry for the manual group (the current revision), not one per
        # revision file.
        assert matched.count("widget_manual_v2.txt") <= 1
        assert "widget_manual_v1.txt" not in matched
