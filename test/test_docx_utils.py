#!/usr/bin/env python3
"""Tests for chunksilo.docx_utils module."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from docx import Document

from chunksilo.docx_utils import (
    _convert_doc_to_docx,
    _is_heading_style,
    _parse_heading_level,
    split_docx_into_heading_documents,
)

# =============================================================================
# Tests for _parse_heading_level
# =============================================================================


class TestParseHeadingLevel:
    def test_heading_1(self):
        assert _parse_heading_level("Heading 1") == 1

    def test_heading_2(self):
        assert _parse_heading_level("Heading 2") == 2

    def test_heading_9(self):
        assert _parse_heading_level("Heading 9") == 9

    def test_heading_no_number(self):
        assert _parse_heading_level("Heading") == 1

    def test_none_returns_default(self):
        assert _parse_heading_level(None) == 1

    def test_empty_string_returns_default(self):
        assert _parse_heading_level("") == 1

    def test_non_heading_style(self):
        assert _parse_heading_level("Normal") == 1

    def test_heading_with_extra_whitespace(self):
        assert _parse_heading_level("Heading  3") == 3

    def test_heading_non_numeric_suffix(self):
        assert _parse_heading_level("Heading abc") == 1


# =============================================================================
# Tests for _is_heading_style
# =============================================================================


class TestIsHeadingStyle:
    def test_standard_heading(self):
        assert _is_heading_style("Heading 1") is True

    def test_lowercase_heading(self):
        assert _is_heading_style("heading 2") is True

    def test_heading_no_number(self):
        assert _is_heading_style("Heading") is True

    def test_contains_heading(self):
        assert _is_heading_style("Custom Heading Style") is True

    def test_normal_style(self):
        assert _is_heading_style("Normal") is False

    def test_empty_string(self):
        assert _is_heading_style("") is False

    def test_title_style(self):
        assert _is_heading_style("Title") is False


# =============================================================================
# Tests for _convert_doc_to_docx
# =============================================================================


class TestConvertDocToDocx:
    @patch("chunksilo.docx_utils._get_doc_temp_dir")
    @patch("shutil.which", return_value=None)
    def test_libreoffice_not_found(self, mock_which, mock_temp_dir):
        result = _convert_doc_to_docx(Path("/fake/test.doc"))
        assert result is None

    @patch("chunksilo.docx_utils._get_doc_temp_dir")
    @patch("subprocess.run")
    @patch("shutil.which")
    def test_conversion_success(self, mock_which, mock_run, mock_temp_dir, tmp_path):
        mock_which.side_effect = lambda p: "/usr/bin/soffice" if p == "/usr/bin/soffice" else None
        mock_run.return_value = MagicMock(returncode=0)
        mock_temp_dir.return_value = tmp_path

        # Create the expected output file
        output_file = tmp_path / "test.docx"
        output_file.touch()

        result = _convert_doc_to_docx(Path("/fake/test.doc"))
        assert result == output_file

    @patch("chunksilo.docx_utils._get_doc_temp_dir")
    @patch("subprocess.run")
    @patch("shutil.which")
    def test_conversion_timeout(self, mock_which, mock_run, mock_temp_dir, tmp_path):
        mock_which.side_effect = lambda p: "soffice" if p == "soffice" else None
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="soffice", timeout=60)
        mock_temp_dir.return_value = tmp_path

        result = _convert_doc_to_docx(Path("/fake/test.doc"))
        assert result is None

    @patch("chunksilo.docx_utils._get_doc_temp_dir")
    @patch("subprocess.run")
    @patch("shutil.which")
    def test_conversion_nonzero_exit(self, mock_which, mock_run, mock_temp_dir, tmp_path):
        mock_which.side_effect = lambda p: "/usr/bin/soffice" if p == "/usr/bin/soffice" else None
        mock_run.return_value = MagicMock(returncode=1, stderr=b"error")
        mock_temp_dir.return_value = tmp_path

        result = _convert_doc_to_docx(Path("/fake/test.doc"))
        assert result is None


# =============================================================================
# Helpers for creating DOCX fixtures
# =============================================================================


def _create_docx(paragraphs: list[tuple[str, str]], tmp_path: Path) -> Path:
    """Create a DOCX file with the given paragraphs.

    Each entry is (style_name, text). Returns the path to the file.
    """
    doc = Document()
    for style_name, text in paragraphs:
        doc.add_paragraph(text, style=style_name)
    path = tmp_path / "test.docx"
    doc.save(str(path))
    return path


# =============================================================================
# Tests for split_docx_into_heading_documents
# =============================================================================


class TestSplitDocxIntoHeadingDocuments:
    def test_multiple_headings_different_levels(self, tmp_path):
        docx_path = _create_docx([
            ("Heading 1", "Chapter One"),
            ("Normal", "Body text under chapter one."),
            ("Heading 2", "Section A"),
            ("Normal", "Body text under section A."),
            ("Heading 1", "Chapter Two"),
            ("Normal", "Body text under chapter two."),
        ], tmp_path)

        docs = split_docx_into_heading_documents(docx_path)

        assert len(docs) == 3
        assert docs[0].metadata["heading"] == "Chapter One"
        assert docs[0].metadata["heading_level"] == 1
        assert "Body text under chapter one." in docs[0].text

        assert docs[1].metadata["heading"] == "Section A"
        assert docs[1].metadata["heading_level"] == 2
        # heading_path should show hierarchy
        assert docs[1].metadata["heading_path"] == ["Chapter One", "Section A"]

        assert docs[2].metadata["heading"] == "Chapter Two"

    def test_no_headings_fallback_to_single_document(self, tmp_path):
        docx_path = _create_docx([
            ("Normal", "Just some plain text."),
            ("Normal", "More plain text."),
        ], tmp_path)

        docs = split_docx_into_heading_documents(docx_path)

        assert len(docs) == 1
        assert docs[0].metadata["heading"] is None
        assert "Just some plain text." in docs[0].text

    def test_corrupted_file_returns_empty(self, tmp_path):
        bad_file = tmp_path / "corrupt.docx"
        bad_file.write_bytes(b"this is not a docx file")

        docs = split_docx_into_heading_documents(bad_file)
        assert docs == []

    def test_metadata_includes_file_info(self, tmp_path):
        docx_path = _create_docx([
            ("Heading 1", "Test Heading"),
            ("Normal", "Some body text."),
        ], tmp_path)

        docs = split_docx_into_heading_documents(docx_path)

        assert len(docs) == 1
        meta = docs[0].metadata
        assert meta["file_name"] == "test.docx"
        assert meta["file_path"] == str(docx_path)
        assert "creation_date" in meta
        assert "last_modified_date" in meta

    def test_heading_store_receives_headings(self, tmp_path):
        docx_path = _create_docx([
            ("Heading 1", "H1"),
            ("Normal", "Body."),
            ("Heading 2", "H2"),
            ("Normal", "More body."),
        ], tmp_path)

        mock_store = MagicMock()
        split_docx_into_heading_documents(docx_path, heading_store=mock_store)

        mock_store.set_headings.assert_called_once()
        args = mock_store.set_headings.call_args
        assert args[0][0] == str(docx_path)
        headings = args[0][1]
        assert len(headings) == 2
        assert headings[0]["text"] == "H1"
        assert headings[0]["level"] == 1
        assert headings[1]["text"] == "H2"
        assert headings[1]["level"] == 2

    def test_excluded_metadata_keys_passed_through(self, tmp_path):
        docx_path = _create_docx([
            ("Heading 1", "Test"),
            ("Normal", "Body."),
        ], tmp_path)

        docs = split_docx_into_heading_documents(
            docx_path,
            excluded_embed_metadata_keys=["file_path"],
            excluded_llm_metadata_keys=["source"],
        )

        assert docs[0].excluded_embed_metadata_keys == ["file_path"]
        assert docs[0].excluded_llm_metadata_keys == ["source"]

    def test_empty_body_heading_skipped(self, tmp_path):
        docx_path = _create_docx([
            ("Heading 1", "Has Body"),
            ("Normal", "Content here."),
            ("Heading 1", "No Body"),
            ("Heading 1", "Also Has Body"),
            ("Normal", "More content."),
        ], tmp_path)

        docs = split_docx_into_heading_documents(docx_path)

        headings = [d.metadata["heading"] for d in docs]
        assert "Has Body" in headings
        assert "Also Has Body" in headings
        assert "No Body" not in headings
