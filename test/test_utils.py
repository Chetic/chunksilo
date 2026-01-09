#!/usr/bin/env python3
"""Unit tests for utility functions in mcp_server.py and ingest.py.

These tests cover pure utility functions that can be tested in isolation
without mocking external dependencies.
"""

import sys
from pathlib import Path
from datetime import datetime

import pytest

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_index.core.schema import TextNode, NodeWithScore


# =============================================================================
# Tests for _reciprocal_rank_fusion (mcp_server.py)
# =============================================================================

class TestReciprocalRankFusion:
    """Tests for _reciprocal_rank_fusion in mcp_server.py"""

    def _create_node(self, text: str, node_id: str, score: float = 0.5) -> NodeWithScore:
        """Helper to create NodeWithScore objects."""
        node = TextNode(text=text, id_=node_id)
        return NodeWithScore(node=node, score=score)

    def test_single_list(self):
        """RRF with a single list returns items with RRF scores."""
        from mcp_server import _reciprocal_rank_fusion

        nodes = [
            self._create_node("First", "a", 0.9),
            self._create_node("Second", "b", 0.8),
            self._create_node("Third", "c", 0.7),
        ]
        result = _reciprocal_rank_fusion([nodes])

        assert len(result) == 3
        # First item should have highest RRF score (1/(60+1) = ~0.0164)
        # All items should have RRF scores based on rank
        assert result[0].node.id_ == "a"
        assert result[0].score > result[1].score > result[2].score

    def test_two_lists_no_overlap(self):
        """RRF with disjoint lists combines them."""
        from mcp_server import _reciprocal_rank_fusion

        list1 = [self._create_node("A", "a", 0.9)]
        list2 = [self._create_node("B", "b", 0.8)]

        result = _reciprocal_rank_fusion([list1, list2])

        assert len(result) == 2
        ids = {n.node.id_ for n in result}
        assert ids == {"a", "b"}

    def test_two_lists_with_overlap(self):
        """RRF with overlapping items boosts shared items."""
        from mcp_server import _reciprocal_rank_fusion

        # Both lists have item "a" at position 1
        list1 = [
            self._create_node("Shared", "a", 0.9),
            self._create_node("Only in 1", "b", 0.8),
        ]
        list2 = [
            self._create_node("Shared", "a", 0.9),
            self._create_node("Only in 2", "c", 0.8),
        ]

        result = _reciprocal_rank_fusion([list1, list2])

        # Item "a" should be first due to appearing in both lists
        assert result[0].node.id_ == "a"
        # Item "a" score should be ~2x the score of unique items
        assert result[0].score > result[1].score

    def test_empty_lists(self):
        """RRF with empty lists returns empty result."""
        from mcp_server import _reciprocal_rank_fusion

        result = _reciprocal_rank_fusion([])
        assert result == []

        result = _reciprocal_rank_fusion([[], []])
        assert result == []

    def test_k_parameter_effect(self):
        """Different k values affect score distribution."""
        from mcp_server import _reciprocal_rank_fusion

        nodes = [self._create_node("Item", "a", 0.9)]

        # With k=60 (default), score = 1/(60+1) ≈ 0.0164
        result_60 = _reciprocal_rank_fusion([nodes], k=60)
        # With k=1, score = 1/(1+1) = 0.5
        result_1 = _reciprocal_rank_fusion([nodes], k=1)

        # Higher k means lower individual scores
        assert result_1[0].score > result_60[0].score


# =============================================================================
# Tests for tokenize_filename (ingest.py)
# =============================================================================

class TestTokenizeFilename:
    """Tests for tokenize_filename in ingest.py"""

    def test_underscore_delimiter(self):
        """Underscore separates tokens."""
        from ingest import tokenize_filename

        result = tokenize_filename("cpp_styleguide.md")
        assert "cpp" in result
        assert "styleguide" in result
        assert "md" in result

    def test_hyphen_delimiter(self):
        """Hyphen separates tokens."""
        from ingest import tokenize_filename

        result = tokenize_filename("API-Reference-v2.pdf")
        # Should be lowercase
        assert "api" in result
        assert "reference" in result
        assert "v2" in result
        assert "pdf" in result

    def test_camel_case(self):
        """CamelCase is split into tokens."""
        from ingest import tokenize_filename

        result = tokenize_filename("CamelCaseDoc.docx")
        assert "camel" in result
        assert "case" in result
        assert "doc" in result
        assert "docx" in result

    def test_mixed_delimiters(self):
        """Mixed delimiters all separate tokens."""
        from ingest import tokenize_filename

        result = tokenize_filename("Mixed_Case-File.Name.md")
        assert "mixed" in result
        assert "case" in result
        assert "file" in result
        assert "name" in result
        assert "md" in result

    def test_no_extension(self):
        """Files without extension are tokenized."""
        from ingest import tokenize_filename

        result = tokenize_filename("README")
        assert "readme" in result

    def test_multiple_dots(self):
        """Multiple dots in filename are handled."""
        from ingest import tokenize_filename

        result = tokenize_filename("file.tar.gz")
        # Extension handling may vary - just ensure no crash
        assert len(result) > 0


# =============================================================================
# Tests for _filter_nodes_by_date (mcp_server.py)
# =============================================================================

class TestFilterNodesByDate:
    """Tests for _filter_nodes_by_date in mcp_server.py"""

    def _create_node_with_date(self, node_id: str, date: str) -> NodeWithScore:
        """Helper to create NodeWithScore with date metadata."""
        node = TextNode(
            text=f"Content for {node_id}",
            id_=node_id,
            metadata={"creation_date": date}
        )
        return NodeWithScore(node=node, score=0.5)

    def test_no_filters_returns_all(self):
        """No date filters returns all nodes."""
        from mcp_server import _filter_nodes_by_date

        nodes = [
            self._create_node_with_date("a", "2024-01-15"),
            self._create_node_with_date("b", "2024-06-15"),
        ]

        result = _filter_nodes_by_date(nodes, None, None)
        assert len(result) == 2

    def test_date_from_filter(self):
        """Filters out documents before date_from."""
        from mcp_server import _filter_nodes_by_date

        nodes = [
            self._create_node_with_date("old", "2024-01-15"),
            self._create_node_with_date("new", "2024-06-15"),
        ]

        result = _filter_nodes_by_date(nodes, date_from="2024-03-01", date_to=None)

        assert len(result) == 1
        assert result[0].node.id_ == "new"

    def test_date_to_filter(self):
        """Filters out documents after date_to."""
        from mcp_server import _filter_nodes_by_date

        nodes = [
            self._create_node_with_date("old", "2024-01-15"),
            self._create_node_with_date("new", "2024-06-15"),
        ]

        result = _filter_nodes_by_date(nodes, date_from=None, date_to="2024-03-01")

        assert len(result) == 1
        assert result[0].node.id_ == "old"

    def test_date_range_filter(self):
        """Both filters work together."""
        from mcp_server import _filter_nodes_by_date

        nodes = [
            self._create_node_with_date("jan", "2024-01-15"),
            self._create_node_with_date("mar", "2024-03-15"),
            self._create_node_with_date("jun", "2024-06-15"),
        ]

        result = _filter_nodes_by_date(nodes, date_from="2024-02-01", date_to="2024-04-01")

        assert len(result) == 1
        assert result[0].node.id_ == "mar"

    def test_no_date_metadata_passes(self):
        """Documents without dates pass through (backward compatibility)."""
        from mcp_server import _filter_nodes_by_date

        node_no_date = TextNode(text="No date", id_="no_date", metadata={})
        nodes = [NodeWithScore(node=node_no_date, score=0.5)]

        result = _filter_nodes_by_date(nodes, date_from="2024-01-01", date_to="2024-12-31")

        assert len(result) == 1


# =============================================================================
# Tests for _apply_recency_boost (mcp_server.py)
# =============================================================================

class TestApplyRecencyBoost:
    """Tests for _apply_recency_boost in mcp_server.py"""

    def _create_node_with_date(self, node_id: str, date: str, score: float = 0.5) -> NodeWithScore:
        """Helper to create NodeWithScore with date metadata."""
        node = TextNode(
            text=f"Content for {node_id}",
            id_=node_id,
            metadata={"creation_date": date}
        )
        return NodeWithScore(node=node, score=score)

    def test_zero_boost_weight(self):
        """boost_weight=0 returns original scores."""
        from mcp_server import _apply_recency_boost

        nodes = [
            self._create_node_with_date("a", "2024-01-15", 0.9),
            self._create_node_with_date("b", "2024-06-15", 0.8),
        ]

        result = _apply_recency_boost(nodes, boost_weight=0.0)

        # With zero boost, order should be unchanged
        assert result[0].node.id_ == "a"
        assert result[1].node.id_ == "b"

    def test_no_date_unchanged(self):
        """Documents without dates use base score only."""
        from mcp_server import _apply_recency_boost

        node_no_date = TextNode(text="No date", id_="no_date", metadata={})
        nodes = [NodeWithScore(node=node_no_date, score=0.5)]

        result = _apply_recency_boost(nodes, boost_weight=0.5)

        # Should have result and not crash
        assert len(result) == 1

    def test_empty_list(self):
        """Empty list returns empty list."""
        from mcp_server import _apply_recency_boost

        result = _apply_recency_boost([], boost_weight=0.5)
        assert result == []


# =============================================================================
# Tests for _parse_date (mcp_server.py)
# =============================================================================

class TestParseDate:
    """Tests for _parse_date in mcp_server.py"""

    def test_valid_date_format(self):
        """Valid YYYY-MM-DD parses correctly."""
        from mcp_server import _parse_date

        result = _parse_date("2024-01-15")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_invalid_format(self):
        """Invalid format returns None."""
        from mcp_server import _parse_date

        result = _parse_date("01/15/2024")
        assert result is None

    def test_empty_string(self):
        """Empty string returns None."""
        from mcp_server import _parse_date

        result = _parse_date("")
        assert result is None

    def test_none_input(self):
        """None input returns None."""
        from mcp_server import _parse_date

        result = _parse_date(None)
        assert result is None


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
