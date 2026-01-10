"""Data source abstractions for document ingestion."""

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List

from llama_index.core import SimpleDirectoryReader

from opd_mcp.config import EXCLUDED_EMBED_METADATA_KEYS, EXCLUDED_LLM_METADATA_KEYS
from opd_mcp.storage import FileInfo, get_heading_store
from opd_mcp.utils.text import compute_line_offsets
from opd_mcp.ingestion.extractors import (
    extract_markdown_headings,
    extract_pdf_headings_from_outline,
)
from opd_mcp.ingestion.docx import split_docx_into_heading_documents

logger = logging.getLogger(__name__)

# Import Document type from llama_index
from llama_index.core import Document as LlamaIndexDocument


class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def iter_files(self) -> Iterator[FileInfo]:
        """Yield FileInfo for each file in the source."""
        pass

    @abstractmethod
    def load_file(self, file_info: FileInfo) -> List[LlamaIndexDocument]:
        """Load and return documents for a given file."""
        pass


class LocalFileSystemSource(DataSource):
    """Implementation of DataSource for the local file system."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def iter_files(self) -> Iterator[FileInfo]:
        """Iterate over all supported files in the base directory."""
        extensions = {".pdf", ".md", ".txt", ".docx"}
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if Path(file).suffix.lower() in extensions:
                    file_path = Path(root) / file
                    yield self._create_file_info(file_path)

    def _create_file_info(self, file_path: Path) -> FileInfo:
        """Create a FileInfo object for a file."""
        # Create a hash of the file content
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)

        return FileInfo(
            path=str(file_path.absolute()),
            hash=hasher.hexdigest(),
            last_modified=file_path.stat().st_mtime,
        )

    def load_file(self, file_info: FileInfo) -> List[LlamaIndexDocument]:
        """Load a file and return its documents."""
        file_path = Path(file_info.path)

        if file_path.suffix.lower() == ".docx":
            return split_docx_into_heading_documents(file_path)
        else:
            reader = SimpleDirectoryReader(
                input_files=[str(file_path)],
            )
            docs = reader.load_data()

            # Ensure dates are visible to LLM (remove from exclusion list)
            for doc in docs:
                if hasattr(doc, "excluded_llm_metadata_keys") and doc.excluded_llm_metadata_keys:
                    doc.excluded_llm_metadata_keys = [
                        k
                        for k in doc.excluded_llm_metadata_keys
                        if k not in ("creation_date", "last_modified_date")
                    ]

            # Add line offsets for text-based files (markdown, txt)
            if file_path.suffix.lower() in {".md", ".txt"}:
                for doc in docs:
                    text = doc.get_content()
                    line_offsets = compute_line_offsets(text)
                    doc.metadata["line_offsets"] = line_offsets

                    # Extract headings for Markdown and store separately
                    if file_path.suffix.lower() == ".md":
                        headings = extract_markdown_headings(text)
                        get_heading_store().set_headings(str(file_path), headings)

            # Extract headings for PDF files and store separately
            if file_path.suffix.lower() == ".pdf":
                headings = extract_pdf_headings_from_outline(file_path)
                get_heading_store().set_headings(str(file_path), headings)

            # Apply metadata exclusions
            for doc in docs:
                doc.excluded_embed_metadata_keys = EXCLUDED_EMBED_METADATA_KEYS
                doc.excluded_llm_metadata_keys = EXCLUDED_LLM_METADATA_KEYS

            return docs
