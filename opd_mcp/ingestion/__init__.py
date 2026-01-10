"""Ingestion modules for document processing and indexing."""

from opd_mcp.ingestion.source import DataSource, LocalFileSystemSource
from opd_mcp.ingestion.extractors import (
    extract_markdown_headings,
    extract_pdf_headings_from_outline,
)
from opd_mcp.ingestion.docx import split_docx_into_heading_documents
from opd_mcp.ingestion.bm25 import build_bm25_index

__all__ = [
    "DataSource",
    "LocalFileSystemSource",
    "extract_markdown_headings",
    "extract_pdf_headings_from_outline",
    "split_docx_into_heading_documents",
    "build_bm25_index",
]
