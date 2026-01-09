#!/usr/bin/env python3
"""Tests for error handling and edge cases in mcp_server.py and ingest.py.

These tests verify that the system handles malformed inputs, missing data,
and error conditions gracefully without crashing.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Tests for empty/invalid query handling
# =============================================================================

class TestEmptyQueries:
    """Tests for handling empty or whitespace-only queries."""

    def test_preprocess_empty_query(self):
        """Empty query returns empty or original."""
        from mcp_server import _preprocess_query

        result = _preprocess_query("")
        # Should not crash, return empty or stripped
        assert isinstance(result, str)

    def test_preprocess_whitespace_query(self):
        """Whitespace-only query is handled."""
        from mcp_server import _preprocess_query

        result = _preprocess_query("   ")
        # Should strip whitespace
        assert result.strip() == ""


# =============================================================================
# Tests for invalid date handling
# =============================================================================

class TestInvalidDates:
    """Tests for handling invalid date inputs."""

    def test_parse_date_malformed(self):
        """Malformed date string returns None."""
        from mcp_server import _parse_date

        # Various malformed formats
        assert _parse_date("not-a-date") is None
        assert _parse_date("2024/01/15") is None  # Wrong separator
        assert _parse_date("15-01-2024") is None  # Wrong order
        assert _parse_date("2024-13-01") is None  # Invalid month

    def test_parse_date_partial(self):
        """Partial date string returns None."""
        from mcp_server import _parse_date

        assert _parse_date("2024-01") is None  # Missing day
        assert _parse_date("2024") is None  # Just year

    def test_filter_nodes_invalid_dates(self):
        """Filter with invalid date strings doesn't crash."""
        from mcp_server import _filter_nodes_by_date
        from llama_index.core.schema import TextNode, NodeWithScore

        node = TextNode(
            text="Test",
            id_="test",
            metadata={"creation_date": "2024-06-15"}
        )
        nodes = [NodeWithScore(node=node, score=0.5)]

        # Invalid date formats should be handled gracefully
        # They'll fail to parse and return None, so no filtering happens
        result = _filter_nodes_by_date(nodes, "not-a-date", "also-not-a-date")

        # Should return nodes (no filtering when dates can't be parsed)
        assert len(result) == 1


# =============================================================================
# Tests for missing metadata handling
# =============================================================================

class TestMissingMetadata:
    """Tests for handling nodes with missing or incomplete metadata."""

    def test_build_heading_path_missing_position(self):
        """Heading without position is handled."""
        from mcp_server import _build_heading_path

        # Heading missing 'position' key
        headings = [{"text": "Chapter 1"}]  # No 'position'
        heading_text, path = _build_heading_path(headings, 50)

        # Should not crash - behavior may vary
        assert isinstance(path, list)

    def test_char_offset_to_line_missing_offsets(self):
        """Missing line_offsets returns None."""
        from mcp_server import _char_offset_to_line

        assert _char_offset_to_line(100, None) is None
        assert _char_offset_to_line(100, []) is None

    def test_recency_boost_missing_date(self):
        """Node without date metadata gets base score."""
        from mcp_server import _apply_recency_boost
        from llama_index.core.schema import TextNode, NodeWithScore

        # Node with no date metadata
        node = TextNode(text="No date", id_="no_date", metadata={})
        nodes = [NodeWithScore(node=node, score=0.7)]

        result = _apply_recency_boost(nodes, boost_weight=0.5)

        # Should not crash and return the node
        assert len(result) == 1


# =============================================================================
# Tests for RRF edge cases
# =============================================================================

class TestRRFEdgeCases:
    """Tests for edge cases in reciprocal rank fusion."""

    def test_rrf_with_none_scores(self):
        """Nodes with None scores are handled."""
        from mcp_server import _reciprocal_rank_fusion
        from llama_index.core.schema import TextNode, NodeWithScore

        node = TextNode(text="Test", id_="test")
        # NodeWithScore with None score
        nodes = [NodeWithScore(node=node, score=None)]

        result = _reciprocal_rank_fusion([nodes])

        # Should not crash, result should have RRF score
        assert len(result) == 1
        assert result[0].score is not None

    def test_rrf_duplicate_ids_same_list(self):
        """Duplicate IDs in same list are handled."""
        from mcp_server import _reciprocal_rank_fusion
        from llama_index.core.schema import TextNode, NodeWithScore

        # Same ID appearing twice in one list (shouldn't happen but test anyway)
        nodes = [
            NodeWithScore(node=TextNode(text="First", id_="dup"), score=0.9),
            NodeWithScore(node=TextNode(text="Second", id_="dup"), score=0.8),
        ]

        result = _reciprocal_rank_fusion([nodes])

        # Should handle gracefully - likely keeps one version
        assert len(result) >= 1


# =============================================================================
# Tests for tokenize_filename edge cases
# =============================================================================

class TestTokenizeFilenameEdgeCases:
    """Tests for edge cases in filename tokenization."""

    def test_empty_filename(self):
        """Empty filename returns empty or minimal tokens."""
        from ingest import tokenize_filename

        result = tokenize_filename("")
        # Should not crash
        assert isinstance(result, list)

    def test_only_extension(self):
        """Filename that is just an extension."""
        from ingest import tokenize_filename

        result = tokenize_filename(".gitignore")
        # Should return something reasonable
        assert isinstance(result, list)
        assert len(result) > 0

    def test_unicode_filename(self):
        """Unicode characters in filename are handled."""
        from ingest import tokenize_filename

        result = tokenize_filename("文档.pdf")
        # Should not crash
        assert isinstance(result, list)

    def test_very_long_filename(self):
        """Very long filename is handled."""
        from ingest import tokenize_filename

        long_name = "a" * 500 + ".txt"
        result = tokenize_filename(long_name)
        # Should not crash
        assert isinstance(result, list)


# =============================================================================
# Tests for compute_line_offsets edge cases
# =============================================================================

class TestComputeLineOffsetsEdgeCases:
    """Tests for edge cases in line offset computation."""

    def test_binary_content(self):
        """Text with binary-like content doesn't crash."""
        from ingest import _compute_line_offsets

        # Text with null bytes and other binary chars (as string)
        text = "Line1\nLine2\x00Line3\n"
        result = _compute_line_offsets(text)

        # Should handle gracefully
        assert isinstance(result, list)
        assert len(result) > 0

    def test_only_newlines(self):
        """Text that is only newlines."""
        from ingest import _compute_line_offsets

        text = "\n\n\n"
        result = _compute_line_offsets(text)

        # Should have offset for each line
        assert isinstance(result, list)
        assert len(result) == 4  # Start + 3 newlines


# =============================================================================
# Tests for IngestionState edge cases
# =============================================================================

class TestIngestionStateEdgeCases:
    """Tests for edge cases in IngestionState database handling."""

    def test_state_creates_directory(self, tmp_path):
        """IngestionState creates parent directories if needed."""
        from ingest import IngestionState

        # Path in non-existent subdirectory
        db_path = tmp_path / "subdir" / "deeper" / "state.db"

        # Should create directories and initialize DB
        state = IngestionState(db_path)

        assert db_path.exists()
        # Verify it's a valid SQLite DB
        files = state.get_all_files()
        assert isinstance(files, dict)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
