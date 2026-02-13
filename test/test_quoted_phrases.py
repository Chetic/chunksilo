"""Tests for quoted phrase exact-match search functionality."""
import pytest
from unittest.mock import MagicMock

from llama_index.core.schema import TextNode, NodeWithScore


# =============================================================================
# Tests for _extract_quoted_phrases
# =============================================================================

class TestExtractQuotedPhrases:
    """Tests for extracting quoted phrases from search queries."""

    def test_single_quoted_phrase(self):
        """Single quoted phrase is extracted, words kept in cleaned query."""
        from chunksilo.search import _extract_quoted_phrases

        phrases, cleaned = _extract_quoted_phrases('find "exact match" here')
        assert phrases == ["exact match"]
        assert cleaned == "find exact match here"

    def test_multiple_quoted_phrases(self):
        """Multiple quoted phrases are all extracted."""
        from chunksilo.search import _extract_quoted_phrases

        phrases, cleaned = _extract_quoted_phrases('"phrase one" and "phrase two"')
        assert phrases == ["phrase one", "phrase two"]
        assert cleaned == "phrase one and phrase two"

    def test_no_quotes(self):
        """Query without quotes returns empty phrases list."""
        from chunksilo.search import _extract_quoted_phrases

        phrases, cleaned = _extract_quoted_phrases("normal query")
        assert phrases == []
        assert cleaned == "normal query"

    def test_empty_quoted_string(self):
        """Empty quotes are ignored."""
        from chunksilo.search import _extract_quoted_phrases

        phrases, cleaned = _extract_quoted_phrases('test "" here')
        assert phrases == []

    def test_entire_query_quoted(self):
        """Entire query in quotes extracts as single phrase."""
        from chunksilo.search import _extract_quoted_phrases

        phrases, cleaned = _extract_quoted_phrases('"full query"')
        assert phrases == ["full query"]
        assert cleaned == "full query"

    def test_unmatched_quote(self):
        """Unmatched quote is left as-is."""
        from chunksilo.search import _extract_quoted_phrases

        phrases, cleaned = _extract_quoted_phrases('test "unmatched')
        assert phrases == []
        assert cleaned == 'test "unmatched'

    def test_whitespace_only_phrase(self):
        """Whitespace-only quoted phrase is ignored."""
        from chunksilo.search import _extract_quoted_phrases

        phrases, cleaned = _extract_quoted_phrases('test "   " here')
        assert phrases == []


# =============================================================================
# Tests for _docstore_phrase_search
# =============================================================================

class TestDocstorePhraseSearch:
    """Tests for scanning docstore for exact phrase matches."""

    def _make_index_with_nodes(self, texts):
        """Helper to create a mock index with docstore containing given texts."""
        index = MagicMock()
        docs = {}
        for i, text in enumerate(texts):
            node = TextNode(
                text=text,
                metadata={"file_path": f"/path/file{i}.txt", "file_name": f"file{i}.txt"},
                id_=f"node_{i}",
            )
            docs[f"doc_{i}"] = node
        index.docstore.docs.items.return_value = list(docs.items())
        return index

    def test_single_phrase_found(self):
        """Finds chunks containing the required phrase."""
        from chunksilo.search import _docstore_phrase_search

        index = self._make_index_with_nodes([
            "This document mentions article-123 in detail.",
            "This document has no matching content.",
            "Another reference to article-123 here.",
        ])

        results = _docstore_phrase_search(index, ["article-123"])
        assert len(results) == 2
        assert all(isinstance(r, NodeWithScore) for r in results)
        assert all(r.score == 0.0 for r in results)

    def test_phrase_not_found(self):
        """Returns empty list when phrase is not in any chunk."""
        from chunksilo.search import _docstore_phrase_search

        index = self._make_index_with_nodes([
            "Some content about Python.",
            "Another document about testing.",
        ])

        results = _docstore_phrase_search(index, ["nonexistent-xyz"])
        assert results == []

    def test_multiple_phrases_all_required(self):
        """All phrases must be present in a chunk for it to match."""
        from chunksilo.search import _docstore_phrase_search

        index = self._make_index_with_nodes([
            "Has phrase-a but not the other.",
            "Has phrase-b but not the other.",
            "Has both phrase-a and phrase-b together.",
        ])

        results = _docstore_phrase_search(index, ["phrase-a", "phrase-b"])
        assert len(results) == 1
        assert "both" in results[0].node.get_content()

    def test_case_insensitive(self):
        """Phrase matching is case-insensitive."""
        from chunksilo.search import _docstore_phrase_search

        index = self._make_index_with_nodes([
            "This mentions ARTICLE-123 in uppercase.",
        ])

        results = _docstore_phrase_search(index, ["article-123"])
        assert len(results) == 1

    def test_result_capping(self):
        """Results are capped at max_results."""
        from chunksilo.search import _docstore_phrase_search

        texts = [f"Document {i} mentions target-phrase here." for i in range(50)]
        index = self._make_index_with_nodes(texts)

        results = _docstore_phrase_search(index, ["target-phrase"], max_results=5)
        assert len(results) == 5

    def test_empty_content_handled(self):
        """Nodes with empty content don't crash."""
        from chunksilo.search import _docstore_phrase_search

        index = MagicMock()
        node = TextNode(text="", metadata={}, id_="empty")
        index.docstore.docs.items.return_value = [("doc_0", node)]

        results = _docstore_phrase_search(index, ["anything"])
        assert results == []


# =============================================================================
# Tests for post-rerank phrase filtering
# =============================================================================

class TestQuotedPhrasePostFilter:
    """Tests for post-rerank filtering of results by required phrases."""

    def _create_node(self, node_id, text):
        """Helper to create a NodeWithScore with given text."""
        node = TextNode(text=text, id_=node_id, metadata={})
        return NodeWithScore(node=node, score=0.5)

    def test_filters_non_matching_nodes(self):
        """Only nodes containing the phrase survive filtering."""
        nodes = [
            self._create_node("a", "This has the target-phrase in it."),
            self._create_node("b", "This does not match at all."),
            self._create_node("c", "Another mention of target-phrase."),
        ]

        phrases = ["target-phrase"]
        lowered = [p.lower() for p in phrases]
        filtered = [
            n for n in nodes
            if all(p in (n.node.get_content() or "").lower() for p in lowered)
        ]

        assert len(filtered) == 2
        assert {n.node.id_ for n in filtered} == {"a", "c"}

    def test_no_phrases_no_filtering(self):
        """Empty phrases list means no filtering."""
        nodes = [
            self._create_node("a", "Any content."),
            self._create_node("b", "More content."),
        ]

        phrases: list[str] = []
        if phrases:
            lowered = [p.lower() for p in phrases]
            nodes = [
                n for n in nodes
                if all(p in (n.node.get_content() or "").lower() for p in lowered)
            ]

        assert len(nodes) == 2

    def test_multiple_phrases_all_required(self):
        """All phrases must be present for a node to survive."""
        nodes = [
            self._create_node("a", "Has alpha but not the other."),
            self._create_node("b", "Has beta but not the other."),
            self._create_node("c", "Has both alpha and beta together."),
        ]

        phrases = ["alpha", "beta"]
        lowered = [p.lower() for p in phrases]
        filtered = [
            n for n in nodes
            if all(p in (n.node.get_content() or "").lower() for p in lowered)
        ]

        assert len(filtered) == 1
        assert filtered[0].node.id_ == "c"

    def test_case_insensitive_filtering(self):
        """Post-filter is case-insensitive."""
        nodes = [
            self._create_node("a", "Contains TARGET-PHRASE in uppercase."),
        ]

        phrases = ["target-phrase"]
        lowered = [p.lower() for p in phrases]
        filtered = [
            n for n in nodes
            if all(p in (n.node.get_content() or "").lower() for p in lowered)
        ]

        assert len(filtered) == 1
