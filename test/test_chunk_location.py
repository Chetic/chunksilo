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
    """Tests for URI building logic in chunk processing"""

    def test_absolute_file_path(self):
        """Test file:// URI generation for absolute paths"""
        file_path = Path("/Users/test/data/document.pdf")
        expected_uri = f"file://{file_path.resolve()}"

        # Simulate the URI building logic
        file_path_obj = Path(str(file_path))
        if file_path_obj.is_absolute():
            source_uri = f"file://{file_path_obj.resolve()}"
        else:
            source_uri = None

        assert source_uri == expected_uri

    def test_relative_file_path(self):
        """Test file:// URI generation for relative paths"""
        with patch.dict(os.environ, {"DATA_DIR": "/Users/test/data"}):
            file_path = "docs/readme.md"
            data_dir = Path(os.getenv("DATA_DIR", "./data"))

            file_path_obj = Path(str(file_path))
            if file_path_obj.is_absolute():
                source_uri = f"file://{file_path_obj.resolve()}"
            else:
                resolved_path = (data_dir / file_path_obj).resolve()
                source_uri = f"file://{resolved_path}"

            assert source_uri.startswith("file://")
            assert "docs/readme.md" in source_uri or "readme.md" in source_uri

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
        file_path_obj = Path(str(file_path))
        source_uri = f"file://{file_path_obj.resolve()}"

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
        file_path_obj = Path(str(file_path))
        source_uri = f"file://{file_path_obj.resolve()}"

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
