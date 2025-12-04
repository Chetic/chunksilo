#!/usr/bin/env python3
"""
Ingestion pipeline for building a RAG index from PDF, DOCX, and Markdown documents.
"""
import os
import sys
import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from docx import Document

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    SimpleDirectoryReader,
    Document as LlamaIndexDocument,
)
from llama_index.embeddings.fastembed import FastEmbedEmbedding

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "./storage"))
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "BAAI/bge-small-en-v1.5")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SimpleProgressBar:
    """Lightweight progress bar using only the standard library."""

    def __init__(self, total: int, desc: str, unit: str = "item", width: int = 30):
        self.total = max(total, 0)
        self.desc = desc
        self.unit = unit
        self.width = width
        self.current = 0
        if self.total > 0:
            self._render()

    def update(self, step: int = 1) -> None:
        if self.total <= 0:
            return
        self.current = min(self.total, self.current + step)
        self._render()
        if self.current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _render(self) -> None:
        progress = self.current / self.total if self.total else 0
        filled = int(self.width * progress)
        bar = "#" * filled + "-" * (self.width - filled)
        sys.stdout.write(
            f"\r{self.desc} [{bar}] {progress * 100:5.1f}% ({self.current}/{self.total} {self.unit}s)"
        )
        sys.stdout.flush()


def _parse_heading_level(style_name: str | None) -> int:
    """Best-effort extraction of a numeric heading level from a DOCX style name."""
    if not style_name:
        return 1
    try:
        if "Heading" in style_name:
            level_str = style_name.replace("Heading", "").strip()
            if level_str:
                return int(level_str)
    except (ValueError, AttributeError):
        pass
    return 1


def split_docx_into_heading_documents(docx_path: Path) -> List[LlamaIndexDocument]:
    """
    Load a DOCX file and split it into one LlamaIndex document **per heading section**.

    Each returned document:
    - text: all paragraphs that belong to a given heading (up to, but not including,
      the next heading of the same or higher level)
    - metadata: includes file + heading information so the MCP server can surface
      section-level citations.
    """
    docs: List[LlamaIndexDocument] = []

    try:
        doc = Document(docx_path)
    except Exception as e:
        logger.warning(f"Failed to open DOCX {docx_path}: {e}")
        return docs

    current_heading: str | None = None
    current_level: int | None = None
    current_body: list[str] = []

    def flush_current():
        """Flush the current heading section into a LlamaIndexDocument."""
        if not current_heading:
            return
        # Join body paragraphs; skip empty sections
        text = "\n".join(line for line in current_body if line is not None).strip()
        if not text:
            return

        metadata = {
            "file_path": str(docx_path),
            "file_name": docx_path.name,
            "source": str(docx_path),
            "heading": current_heading,
            "heading_level": current_level,
        }
        docs.append(LlamaIndexDocument(text=text, metadata=metadata))

    for para in doc.paragraphs:
        style_name = getattr(para.style, "name", "") or ""
        is_heading = (
            style_name.startswith("Heading")
            or style_name.startswith("heading")
            or "Heading" in style_name
        )

        if is_heading and para.text.strip():
            # Starting a new heading section: flush the previous one
            flush_current()

            current_heading = para.text.strip()
            current_level = _parse_heading_level(style_name)
            current_body = []
        else:
            # Regular content: attach to current heading (if any)
            if current_heading is not None:
                current_body.append(para.text)

    # Flush the final section
    flush_current()

    # If no headings were detected, fall back to a single document for the whole file
    if not docs:
        try:
            full_text = "\n".join(p.text for p in doc.paragraphs).strip()
        except Exception:
            full_text = ""

        if full_text:
            metadata = {
                "file_path": str(docx_path),
                "file_name": docx_path.name,
                "source": str(docx_path),
                "heading": None,
                "heading_level": None,
            }
            docs.append(LlamaIndexDocument(text=full_text, metadata=metadata))

    logger.info(
        f"Split DOCX {docx_path} into {len(docs)} heading-based document(s) for indexing"
    )
    return docs


def load_docx_heading_documents(root_dir: Path) -> List[LlamaIndexDocument]:
    """
    Walk the DATA_DIR tree and load all .docx files as heading-based documents.
    """
    docx_paths = list(root_dir.rglob("*.docx"))
    if not docx_paths:
        logger.info("No DOCX files found in data directory; skipping DOCX ingestion.")
        return []

    logger.info(
        f"Found {len(docx_paths)} DOCX file(s); splitting by heading with progress bar..."
    )
    all_docs: List[LlamaIndexDocument] = []
    progress = SimpleProgressBar(len(docx_paths), desc="Processing DOCX files", unit="file")
    for docx_path in docx_paths:
        all_docs.extend(split_docx_into_heading_documents(docx_path))
        progress.update()
    return all_docs


def build_index():
    """Build and persist the vector index from documents."""
    logger.info(f"Starting ingestion from {DATA_DIR}")
    
    # Ensure data directory exists
    if not DATA_DIR.exists():
        logger.error(f"Data directory {DATA_DIR} does not exist!")
        raise FileNotFoundError(f"Data directory {DATA_DIR} does not exist")

    # Discover whether there are any PDF/Markdown files before constructing the reader
    pdf_md_files = list(DATA_DIR.rglob("*.pdf")) + list(DATA_DIR.rglob("*.md"))
    docx_files = list(DATA_DIR.rglob("*.docx"))
    total_files = len(pdf_md_files) + len(docx_files)
    logger.info(
        f"Discovered {len(pdf_md_files)} PDF/Markdown and {len(docx_files)} DOCX file(s) (total: {total_files})."
    )

    docs: List[LlamaIndexDocument] = []

    if pdf_md_files:
        # Initialize reader for non-DOCX formats
        reader = SimpleDirectoryReader(
            input_dir=str(DATA_DIR),
            required_exts=[".pdf", ".md"],
            recursive=True,
        )

        # Load documents (PDF/Markdown)
        logger.info(
            f"Loading {len(pdf_md_files)} PDF/Markdown file(s) into LlamaIndex documents..."
        )
        docs.extend(reader.load_data())
    else:
        logger.info("No PDF/Markdown files found in data directory; skipping PDF/MD ingestion.")

    # Load DOCX documents as one document per heading section
    logger.info("Loading DOCX documents (one chunk per heading)...")
    docx_docs = load_docx_heading_documents(DATA_DIR)
    docs.extend(docx_docs)
    
    if not docs:
        logger.warning(f"No documents found in {DATA_DIR}")
        return
    logger.info(f"Loaded {len(docs)} documents (including DOCX heading chunks)")
    
    # Initialize embedding model
    logger.info(f"Initializing embedding model: {EMB_MODEL_NAME}")
    embed_model = FastEmbedEmbedding(model_name=EMB_MODEL_NAME)
    Settings.embed_model = embed_model

    logger.info(
        "Indexing documents with overall progress bar (progress reflects full document processing)..."
    )
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex([], storage_context=storage_context, embed_model=embed_model)

    progress = SimpleProgressBar(len(docs), desc="Indexing documents", unit="doc")
    for doc in docs:
        index.insert(doc, show_progress=True)
        progress.update()
    
    # Persist index
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Persisting index to {STORAGE_DIR}")
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    
    logger.info(f"Successfully indexed {len(docs)} documents into {STORAGE_DIR}")


if __name__ == "__main__":
    try:
        build_index()
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise

