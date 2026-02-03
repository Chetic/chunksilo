#!/usr/bin/env python3
"""Unit tests for utility functions in chunksilo.py and index.py.

These tests cover pure utility functions that can be tested in isolation
without mocking external dependencies.
"""

import sys
from datetime import datetime

import pytest


from llama_index.core.schema import TextNode, NodeWithScore


# =============================================================================
# Tests for tokenize_filename (index.py)
# =============================================================================

class TestTokenizeFilename:
    """Tests for tokenize_filename in index.py"""

    def test_underscore_delimiter(self):
        """Underscore separates tokens."""
        from chunksilo.index import tokenize_filename

        result = tokenize_filename("cpp_styleguide.md")
        assert "cpp" in result
        assert "styleguide" in result
        assert "md" in result

    def test_hyphen_delimiter(self):
        """Hyphen separates tokens."""
        from chunksilo.index import tokenize_filename

        result = tokenize_filename("API-Reference-v2.pdf")
        # Should be lowercase
        assert "api" in result
        assert "reference" in result
        assert "v2" in result
        assert "pdf" in result

    def test_camel_case(self):
        """CamelCase is split into tokens."""
        from chunksilo.index import tokenize_filename

        result = tokenize_filename("CamelCaseDoc.docx")
        assert "camel" in result
        assert "case" in result
        assert "doc" in result
        assert "docx" in result

    def test_mixed_delimiters(self):
        """Mixed delimiters all separate tokens."""
        from chunksilo.index import tokenize_filename

        result = tokenize_filename("Mixed_Case-File.Name.md")
        assert "mixed" in result
        assert "case" in result
        assert "file" in result
        assert "name" in result
        assert "md" in result

    def test_no_extension(self):
        """Files without extension are tokenized."""
        from chunksilo.index import tokenize_filename

        result = tokenize_filename("README")
        assert "readme" in result

    def test_multiple_dots(self):
        """Multiple dots in filename are handled."""
        from chunksilo.index import tokenize_filename

        result = tokenize_filename("file.tar.gz")
        # Extension handling may vary - just ensure no crash
        assert len(result) > 0


# =============================================================================
# Tests for _filter_nodes_by_date (chunksilo.py)
# =============================================================================

class TestFilterNodesByDate:
    """Tests for _filter_nodes_by_date in chunksilo.py"""

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
        from chunksilo.search import _filter_nodes_by_date

        nodes = [
            self._create_node_with_date("a", "2024-01-15"),
            self._create_node_with_date("b", "2024-06-15"),
        ]

        result = _filter_nodes_by_date(nodes, None, None)
        assert len(result) == 2

    def test_date_from_filter(self):
        """Filters out documents before date_from."""
        from chunksilo.search import _filter_nodes_by_date

        nodes = [
            self._create_node_with_date("old", "2024-01-15"),
            self._create_node_with_date("new", "2024-06-15"),
        ]

        result = _filter_nodes_by_date(nodes, date_from="2024-03-01", date_to=None)

        assert len(result) == 1
        assert result[0].node.id_ == "new"

    def test_date_to_filter(self):
        """Filters out documents after date_to."""
        from chunksilo.search import _filter_nodes_by_date

        nodes = [
            self._create_node_with_date("old", "2024-01-15"),
            self._create_node_with_date("new", "2024-06-15"),
        ]

        result = _filter_nodes_by_date(nodes, date_from=None, date_to="2024-03-01")

        assert len(result) == 1
        assert result[0].node.id_ == "old"

    def test_date_range_filter(self):
        """Both filters work together."""
        from chunksilo.search import _filter_nodes_by_date

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
        from chunksilo.search import _filter_nodes_by_date

        node_no_date = TextNode(text="No date", id_="no_date", metadata={})
        nodes = [NodeWithScore(node=node_no_date, score=0.5)]

        result = _filter_nodes_by_date(nodes, date_from="2024-01-01", date_to="2024-12-31")

        assert len(result) == 1


# =============================================================================
# Tests for _apply_recency_boost (chunksilo.py)
# =============================================================================

class TestApplyRecencyBoost:
    """Tests for _apply_recency_boost in chunksilo.py"""

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
        from chunksilo.search import _apply_recency_boost

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
        from chunksilo.search import _apply_recency_boost

        node_no_date = TextNode(text="No date", id_="no_date", metadata={})
        nodes = [NodeWithScore(node=node_no_date, score=0.5)]

        result = _apply_recency_boost(nodes, boost_weight=0.5)

        # Should have result and not crash
        assert len(result) == 1

    def test_empty_list(self):
        """Empty list returns empty list."""
        from chunksilo.search import _apply_recency_boost

        result = _apply_recency_boost([], boost_weight=0.5)
        assert result == []


# =============================================================================
# Tests for _parse_date (chunksilo.py)
# =============================================================================

class TestParseDate:
    """Tests for _parse_date in chunksilo.py"""

    def test_valid_date_format(self):
        """Valid YYYY-MM-DD parses correctly."""
        from chunksilo.search import _parse_date

        result = _parse_date("2024-01-15")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_invalid_format(self):
        """Invalid format returns None."""
        from chunksilo.search import _parse_date

        result = _parse_date("01/15/2024")
        assert result is None

    def test_empty_string(self):
        """Empty string returns None."""
        from chunksilo.search import _parse_date

        result = _parse_date("")
        assert result is None

    def test_none_input(self):
        """None input returns None."""
        from chunksilo.search import _parse_date

        result = _parse_date(None)
        assert result is None


class TestParseISO8601ToDate:
    """Tests for _parse_iso8601_to_date in search.py"""

    def test_z_suffix(self):
        """Handle Z timezone indicator."""
        from chunksilo.search import _parse_iso8601_to_date

        result = _parse_iso8601_to_date("2024-01-15T10:30:00Z")
        assert result == "2024-01-15"

    def test_z_with_milliseconds(self):
        """Handle Z with fractional seconds."""
        from chunksilo.search import _parse_iso8601_to_date

        result = _parse_iso8601_to_date("2024-01-15T10:30:00.000Z")
        assert result == "2024-01-15"

    def test_timezone_with_colon(self):
        """Handle +00:00 format."""
        from chunksilo.search import _parse_iso8601_to_date

        result = _parse_iso8601_to_date("2024-01-15T10:30:00.000+00:00")
        assert result == "2024-01-15"

    def test_timezone_without_colon_utc(self):
        """Handle +0000 format (Jira)."""
        from chunksilo.search import _parse_iso8601_to_date

        result = _parse_iso8601_to_date("2024-01-15T10:30:00.000+0000")
        assert result == "2024-01-15"

    def test_timezone_without_colon_negative(self):
        """Handle -0500 format."""
        from chunksilo.search import _parse_iso8601_to_date

        result = _parse_iso8601_to_date("2024-01-15T10:30:00.000-0500")
        assert result == "2024-01-15"

    def test_timezone_with_colon_non_utc(self):
        """Handle +05:30 format."""
        from chunksilo.search import _parse_iso8601_to_date

        result = _parse_iso8601_to_date("2024-01-15T10:30:00.000+05:30")
        assert result == "2024-01-15"

    def test_fractional_seconds_variations(self):
        """Handle different fractional second formats."""
        from chunksilo.search import _parse_iso8601_to_date

        # 3 digits
        assert _parse_iso8601_to_date("2024-01-15T10:30:00.000+0000") == "2024-01-15"
        # 2 digits
        assert _parse_iso8601_to_date("2024-01-15T10:30:00.00+0000") == "2024-01-15"
        # 1 digit
        assert _parse_iso8601_to_date("2024-01-15T10:30:00.0+0000") == "2024-01-15"
        # No fractional seconds
        assert _parse_iso8601_to_date("2024-01-15T10:30:00+0000") == "2024-01-15"

    def test_invalid_format(self):
        """Return None for invalid strings."""
        from chunksilo.search import _parse_iso8601_to_date

        result = _parse_iso8601_to_date("not-a-valid-date")
        assert result is None

    def test_empty_string(self):
        """Return None for empty string."""
        from chunksilo.search import _parse_iso8601_to_date

        result = _parse_iso8601_to_date("")
        assert result is None

    def test_none_input(self):
        """Return None for None input."""
        from chunksilo.search import _parse_iso8601_to_date

        result = _parse_iso8601_to_date(None)
        assert result is None

    def test_whitespace_handling(self):
        """Handle leading/trailing whitespace."""
        from chunksilo.search import _parse_iso8601_to_date

        result = _parse_iso8601_to_date("  2024-01-15T10:30:00.000+0000  ")
        assert result == "2024-01-15"

    def test_malformed_iso_string(self):
        """Return None for malformed dates."""
        from chunksilo.search import _parse_iso8601_to_date

        assert _parse_iso8601_to_date("2024-13-45T10:30:00.000+0000") is None
        assert _parse_iso8601_to_date("invalid-iso-timestamp") is None


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
