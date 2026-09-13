#!/usr/bin/env python3
"""Tests for chunk location field generation in search_docs response.

Tests the following location fields:
- uri: file:// URI for local files, full URL for Confluence
- page: page number for PDFs/DOCX
- line: line number for markdown/txt files
- heading_path: section hierarchy from document structure
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from chunksilo import search
from chunksilo.search import _resolve_file_uri

_NO_DIRS = {"indexing": {"directories": []}}



# =============================================================================
# Tests for _compute_line_offsets (index.py)
# =============================================================================

class TestComputeLineOffsets:
    """Tests for _compute_line_offsets in index.py"""

    def test_empty_text(self):
        from chunksilo.index import _compute_line_offsets
        result = _compute_line_offsets("")
        assert result == [0]

    def test_single_line(self):
        from chunksilo.index import _compute_line_offsets
        result = _compute_line_offsets("Hello world")
        assert result == [0]

    def test_multiple_lines(self):
        from chunksilo.index import _compute_line_offsets
        text = "Line 1\nLine 2\nLine 3"
        result = _compute_line_offsets(text)
        # Line 1 starts at 0, Line 2 starts at 7, Line 3 starts at 14
        assert result == [0, 7, 14]

    def test_trailing_newline(self):
        from chunksilo.index import _compute_line_offsets
        text = "Line 1\nLine 2\n"
        result = _compute_line_offsets(text)
        # Line 1 at 0, Line 2 at 7, empty line 3 at 14
        assert result == [0, 7, 14]

    def test_empty_lines(self):
        from chunksilo.index import _compute_line_offsets
        text = "Line 1\n\nLine 3"
        result = _compute_line_offsets(text)
        # Line 1 at 0, empty line 2 at 7, Line 3 at 8
        assert result == [0, 7, 8]


# =============================================================================
# Tests for _char_offset_to_line (chunksilo.py)
# =============================================================================

class TestCharOffsetToLine:
    """Tests for _char_offset_to_line in chunksilo.py"""

    def test_none_offset(self):
        from chunksilo.search import _char_offset_to_line
        result = _char_offset_to_line(None, [0, 10, 20])
        assert result is None

    def test_none_offsets_list(self):
        from chunksilo.search import _char_offset_to_line
        result = _char_offset_to_line(5, None)
        assert result is None

    def test_empty_offsets_list(self):
        from chunksilo.search import _char_offset_to_line
        result = _char_offset_to_line(5, [])
        assert result is None

    def test_first_line(self):
        from chunksilo.search import _char_offset_to_line
        offsets = [0, 10, 20, 30]
        assert _char_offset_to_line(0, offsets) == 1
        assert _char_offset_to_line(5, offsets) == 1
        assert _char_offset_to_line(9, offsets) == 1

    def test_second_line(self):
        from chunksilo.search import _char_offset_to_line
        offsets = [0, 10, 20, 30]
        assert _char_offset_to_line(10, offsets) == 2
        assert _char_offset_to_line(15, offsets) == 2

    def test_last_line(self):
        from chunksilo.search import _char_offset_to_line
        offsets = [0, 10, 20, 30]
        assert _char_offset_to_line(30, offsets) == 4
        assert _char_offset_to_line(35, offsets) == 4

    def test_boundary_cases(self):
        from chunksilo.search import _char_offset_to_line
        offsets = [0, 7, 14, 21]  # "Line 1\nLine 2\nLine 3\nLine 4"
        # Exactly at line starts
        assert _char_offset_to_line(0, offsets) == 1
        assert _char_offset_to_line(7, offsets) == 2
        assert _char_offset_to_line(14, offsets) == 3
        assert _char_offset_to_line(21, offsets) == 4


# =============================================================================
# Tests for _build_heading_path (chunksilo.py)
# =============================================================================

class TestBuildHeadingPath:
    """Tests for _build_heading_path in chunksilo.py"""

    def test_empty_headings(self):
        from chunksilo.search import _build_heading_path
        heading_text, path = _build_heading_path([], 100)
        assert heading_text is None
        assert path == []

    def test_none_char_start(self):
        from chunksilo.search import _build_heading_path
        headings = [{"text": "Intro", "position": 0}]
        heading_text, path = _build_heading_path(headings, None)
        assert heading_text is None
        assert path == []

    def test_single_heading(self):
        from chunksilo.search import _build_heading_path
        headings = [{"text": "Introduction", "position": 0}]
        heading_text, path = _build_heading_path(headings, 50)
        assert heading_text == "Introduction"
        assert path == ["Introduction"]

    def test_multiple_headings_first(self):
        from chunksilo.search import _build_heading_path
        headings = [
            {"text": "Chapter 1", "position": 0},
            {"text": "Chapter 2", "position": 100},
            {"text": "Chapter 3", "position": 200},
        ]
        heading_text, path = _build_heading_path(headings, 50)
        assert heading_text == "Chapter 1"
        assert path == ["Chapter 1"]

    def test_multiple_headings_middle(self):
        from chunksilo.search import _build_heading_path
        headings = [
            {"text": "Chapter 1", "position": 0},
            {"text": "Chapter 2", "position": 100},
            {"text": "Chapter 3", "position": 200},
        ]
        heading_text, path = _build_heading_path(headings, 150)
        assert heading_text == "Chapter 2"
        assert path == ["Chapter 1", "Chapter 2"]

    def test_multiple_headings_last(self):
        from chunksilo.search import _build_heading_path
        headings = [
            {"text": "Chapter 1", "position": 0},
            {"text": "Chapter 2", "position": 100},
            {"text": "Chapter 3", "position": 200},
        ]
        heading_text, path = _build_heading_path(headings, 250)
        assert heading_text == "Chapter 3"
        assert path == ["Chapter 1", "Chapter 2", "Chapter 3"]

    def test_char_start_before_first_heading(self):
        from chunksilo.search import _build_heading_path
        headings = [{"text": "Chapter 1", "position": 100}]
        heading_text, path = _build_heading_path(headings, 50)
        assert heading_text is None
        assert path == []


# =============================================================================
# Tests for URI building logic
# =============================================================================

class TestURIBuilding:
    """Tests for the real URI builder used by the search pipeline.

    These exercise chunksilo.search._resolve_file_uri directly rather than a
    copy of its logic, because the contract that matters cannot be restated:
    it must not touch the filesystem. Every query formats up to fifteen result
    paths, and on a network mount a stat plus a readlink per path component was
    the dominant cost of a search.
    """

    CONFIG = {"indexing": {"directories": ["/Users/test/data"]}}

    @pytest.fixture(autouse=True)
    def _fresh_directory_cache(self, monkeypatch):
        # The configured-directory list is cached module-wide; a test must not
        # inherit another test's config.
        monkeypatch.setattr(search, "_configured_directories_cache", None)

    def test_absolute_file_path(self):
        """An absolute path is used verbatim, not canonicalised."""
        assert (
            _resolve_file_uri("/Users/test/data/document.pdf", self.CONFIG)
            == "file:///Users/test/data/document.pdf"
        )

    def test_relative_file_path(self):
        """A relative path is joined to the first configured directory."""
        assert (
            _resolve_file_uri("docs/readme.md", self.CONFIG)
            == "file:///Users/test/data/docs/readme.md"
        )

    def test_no_filesystem_access(self, monkeypatch):
        """Nothing in URI building may stat, readlink or resolve."""
        def explode(*_args, **_kwargs):
            raise AssertionError("URI building must not touch the filesystem")

        monkeypatch.setattr(os, "stat", explode)
        monkeypatch.setattr(os, "lstat", explode)
        monkeypatch.setattr(os.path, "realpath", explode)
        monkeypatch.setattr(Path, "exists", explode)
        monkeypatch.setattr(Path, "resolve", explode)

        assert (
            _resolve_file_uri("/mnt/docs/spec.pdf", self.CONFIG)
            == "file:///mnt/docs/spec.pdf"
        )

    def test_spaces_and_reserved_characters_are_encoded(self):
        """Real document paths are full of spaces; an unencoded URI is broken."""
        uri = _resolve_file_uri(
            "/srv/docs/Widget Project/10 - Design Notes/Rev A/a#b.docx",
            self.CONFIG,
        )
        assert uri == (
            "file:///srv/docs/Widget%20Project/10%20-%20Design%20Notes/"
            "Rev%20A/a%23b.docx"
        )

    def test_mount_internal_path_is_not_substituted(self):
        """A symlinked mount must keep the name the user knows it by.

        resolve() used to rewrite ~/nas/... into
        /run/user/1000/gvfs/smb-share:server=..., which no application opens.
        """
        assert (
            _resolve_file_uri("/home/alice/nas/Projects/spec.docx", self.CONFIG)
            == "file:///home/alice/nas/Projects/spec.docx"
        )

    def test_empty_path(self):
        assert _resolve_file_uri("", self.CONFIG) is None

    def test_confluence_uri_with_page_id(self):
        """Test Confluence URL generation with page_id"""
        with patch.dict(os.environ, {"CONFLUENCE_URL": "https://wiki.example.com"}):
            confluence_url = os.getenv("CONFLUENCE_URL", "")
            page_id = "12345"

            source_uri = f"{confluence_url.rstrip('/')}/pages/viewpage.action?pageId={page_id}"

            assert source_uri == "https://wiki.example.com/pages/viewpage.action?pageId=12345"

    def test_confluence_uri_without_page_id(self):
        """Test Confluence URL generation without page_id (fallback to title)"""
        from urllib.parse import quote

        with patch.dict(os.environ, {"CONFLUENCE_URL": "https://wiki.example.com"}):
            confluence_url = os.getenv("CONFLUENCE_URL", "")
            title = "Getting Started"

            encoded_title = quote(title.replace(" ", "+"))
            source_uri = f"{confluence_url.rstrip('/')}/spaces/~{encoded_title}"

            assert "wiki.example.com" in source_uri
            assert "Getting" in source_uri


# =============================================================================
# Tests for page number extraction
# =============================================================================

class TestPageNumberExtraction:
    """Tests for page number extraction from metadata"""

    def test_page_label(self):
        """Test extraction from page_label field"""
        metadata = {"page_label": "5"}
        page = metadata.get("page_label") or metadata.get("page_number") or metadata.get("page")
        assert page == "5"

    def test_page_number(self):
        """Test extraction from page_number field"""
        metadata = {"page_number": 10}
        page = metadata.get("page_label") or metadata.get("page_number") or metadata.get("page")
        assert page == 10

    def test_page_field(self):
        """Test extraction from page field"""
        metadata = {"page": 3}
        page = metadata.get("page_label") or metadata.get("page_number") or metadata.get("page")
        assert page == 3

    def test_priority_order(self):
        """Test that page_label takes priority over page_number"""
        metadata = {"page_label": "iv", "page_number": 4}
        page = metadata.get("page_label") or metadata.get("page_number") or metadata.get("page")
        assert page == "iv"

    def test_no_page_info(self):
        """Test when no page info is available"""
        metadata = {"file_name": "doc.md"}
        page = metadata.get("page_label") or metadata.get("page_number") or metadata.get("page")
        assert page is None


# =============================================================================
# Integration tests for full location building
# =============================================================================

class TestLocationIntegration:
    """Integration tests for the full location building logic"""

    def test_pdf_chunk_location(self):
        """Test location fields for a PDF chunk"""
        from chunksilo.search import _build_heading_path

        metadata = {
            "file_path": "/Users/test/data/manual.pdf",
            "page_label": "15",
            "document_headings": [
                {"text": "Introduction", "position": 0},
                {"text": "Installation", "position": 500},
            ],
            "start_char_idx": 600,
        }

        file_path = metadata.get("file_path")
        char_start = metadata.get("start_char_idx")
        headings = metadata.get("document_headings", [])

        # Build URI
        source_uri = _resolve_file_uri(str(file_path), _NO_DIRS)

        # Get page
        page = metadata.get("page_label") or metadata.get("page_number") or metadata.get("page")

        # Get line (None for PDF)
        line = None

        # Get heading path
        _, heading_path = _build_heading_path(headings, char_start)

        location = {
            "uri": source_uri,
            "page": page,
            "line": line,
            "heading_path": heading_path if heading_path else None,
        }

        assert location["uri"].startswith("file://")
        assert location["uri"].endswith("manual.pdf")
        assert location["page"] == "15"
        assert location["line"] is None
        assert location["heading_path"] == ["Introduction", "Installation"]

    def test_markdown_chunk_location(self):
        """Test location fields for a markdown chunk"""
        from chunksilo.search import _char_offset_to_line

        metadata = {
            "file_path": "/Users/test/data/readme.md",
            "line_offsets": [0, 20, 45, 80, 120],  # 5 lines
            "start_char_idx": 50,
            "heading": "Getting Started",
        }

        file_path = metadata.get("file_path")
        char_start = metadata.get("start_char_idx")
        line_offsets = metadata.get("line_offsets")

        # Build URI
        source_uri = _resolve_file_uri(str(file_path), _NO_DIRS)

        # Get page (None for markdown)
        page = None

        # Get line
        line = _char_offset_to_line(char_start, line_offsets)

        # Get heading path
        heading_text = metadata.get("heading")
        heading_path = [heading_text] if heading_text else None

        location = {
            "uri": source_uri,
            "page": page,
            "line": line,
            "heading_path": heading_path,
        }

        assert location["uri"].startswith("file://")
        assert location["uri"].endswith("readme.md")
        assert location["page"] is None
        assert location["line"] == 3  # char 50 is on line 3 (45-79)
        assert location["heading_path"] == ["Getting Started"]

    def test_confluence_chunk_location(self):
        """Test location fields for a Confluence chunk"""
        with patch.dict(os.environ, {"CONFLUENCE_URL": "https://wiki.company.com"}):
            metadata = {
                "source": "Confluence",
                "page_id": "98765",
                "title": "API Documentation",
                "heading": "Authentication",
            }

            # Build URI for Confluence
            confluence_url = os.getenv("CONFLUENCE_URL", "")
            page_id = metadata.get("page_id")
            source_uri = f"{confluence_url.rstrip('/')}/pages/viewpage.action?pageId={page_id}"

            # Get heading path
            heading_text = metadata.get("heading")
            heading_path = [heading_text] if heading_text else None

            location = {
                "uri": source_uri,
                "page": None,
                "line": None,
                "heading_path": heading_path,
            }

            assert location["uri"] == "https://wiki.company.com/pages/viewpage.action?pageId=98765"
            assert location["page"] is None
            assert location["line"] is None
            assert location["heading_path"] == ["Authentication"]
