#!/usr/bin/env python3
"""
MCP server for querying documentation using RAG.
Returns raw document chunks for the calling LLM to synthesize.
"""
import os
import time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load environment variables
load_dotenv()

# Configuration
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "./storage"))
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "5"))
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "BAAI/bge-small-en-v1.5")

# Initialize FastMCP server
mcp = FastMCP("llamaindex-docs-rag")

# Global caches
_index_cache = None
_embed_model_initialized = False


def _get_heading_for_position(headings: list[dict], char_start: int | None) -> dict | None:
    """
    Find the most recent heading before the given character position.
    Returns the heading dict or None if no heading found.
    """
    if not headings or char_start is None:
        return None
    
    current_heading = None
    for heading in headings:
        heading_pos = heading.get('position', 0)
        if heading_pos <= char_start:
            current_heading = heading
        else:
            break
    
    return current_heading


def _build_heading_path(headings: list[dict], char_start: int | None) -> tuple[str | None, list[str]]:
    """
    Build a human-readable heading path (e.g., ["Architecture", "CI/CD", "Deployment"])
    for the given character position within a document.
    """
    if not headings or char_start is None:
        return None, []

    # Find the index of the current heading
    current_idx = None
    for idx, heading in enumerate(headings):
        heading_pos = heading.get("position", 0)
        if heading_pos <= char_start:
            current_idx = idx
        else:
            break

    if current_idx is None:
        return None, []

    # Build path from all headings up to and including the current one
    path = [h.get("text", "") for h in headings[: current_idx + 1] if h.get("text")]
    current_heading_text = path[-1] if path else None
    return current_heading_text, path


def _build_citation(
    metadata: dict[str, Any],
    *,
    heading_text: str | None,
) -> str:
    """
    Return a human-friendly citation string from node metadata.
    Includes page numbers, chapter/heading information, and character ranges when available.
    """

    file_path = (
        metadata.get("file_path")
        or metadata.get("file_name")
        or metadata.get("source")
        or "Unknown source"
    )

    page = metadata.get("page_label") or metadata.get("page_number") or metadata.get("page")

    parts: list[str] = []
    
    # Prefer an explicit heading passed in, but fall back to metadata["heading"]
    heading_for_citation = heading_text or metadata.get("heading")

    # Add section/heading information (most important for DOCX/Markdown files)
    if heading_for_citation:
        parts.append(f'section \"{heading_for_citation}\"')
    
    # Add page number if available
    if page:
        parts.append(f"page {page}")
    
    if parts:
        joined = ", ".join(parts)
        return f"{file_path} ({joined})"

    return str(file_path)


def _ensure_embed_model():
    """
    Ensure the same embedding model used during ingestion is available at query time.

    If this is not set, LlamaIndex falls back to its default (typically an OpenAI
    embedding model), which would require an OPENAI_API_KEY and cause failures
    inside the MCP server.
    """
    global _embed_model_initialized

    if _embed_model_initialized:
        return

    embed_model = HuggingFaceEmbedding(model_name=EMB_MODEL_NAME)
    Settings.embed_model = embed_model
    _embed_model_initialized = True


def load_llamaindex_index():
    """Load the LlamaIndex from storage."""
    global _index_cache

    if _index_cache is not None:
        return _index_cache

    if not STORAGE_DIR.exists():
        raise FileNotFoundError(
            f"Storage directory {STORAGE_DIR} does not exist. "
            "Please run ingest.py first."
        )

    # Make sure the embedding model is configured before using the index so that
    # query embeddings use the same model as ingestion (HuggingFace, not OpenAI).
    _ensure_embed_model()

    storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    index = load_index_from_storage(storage_context)
    _index_cache = index
    return index


@mcp.tool()
async def retrieve_docs(query: str) -> dict[str, Any]:
    """
    Search the local PDF/DOCX/Markdown documentation corpus and return relevant chunks.

    **Mandatory**: When you use information from these chunks in your answer to the user,
    you MUST include source citations. A response that omits citations is incomplete.

    Each returned chunk includes:
    - `text`: Full chunk text content (may start with SOURCE_* helper lines)
    - `citation`: Human-readable citation string (use this field in your answer)
    - `location`: Structured location details (file, page, heading, heading_path)
    - `metadata`: Original document metadata

    Responses from this tool also include a top-level `citations` array listing all unique
    citation strings. Use these values when citing sources in your final answer.
    """
    start_time = time.time()
    
    try:
        # Load index
        index = load_llamaindex_index()

        # Use retriever (no LLM needed - just retrieval)
        retriever = index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)

        # Retrieve relevant chunks
        nodes = retriever.retrieve(query)

        # Format chunks with full content and metadata
        chunks = []
        citation_list = []
        for node in nodes:
            metadata = dict(node.node.metadata or {})
            raw_text = node.node.get_content()

            # Get headings for this chunk/document if available directly from metadata
            file_path = (
                metadata.get("file_path")
                or metadata.get("file_name")
                or metadata.get("source")
            )
            headings = metadata.get("document_headings") or metadata.get("headings") or []

            # Build heading context (current heading + full path) if we have
            # document-level heading structure and character offsets.
            char_start = metadata.get("start_char_idx")
            heading_text = metadata.get("heading")  # may be set directly for per-heading chunks
            heading_path: list[str] = []
            heading_titles: list[str] = []
            if isinstance(headings, list) and headings:
                # Collect all heading titles for this document (normalized)
                heading_titles = [h.get("text", "").strip() for h in headings if isinstance(h, dict) and h.get("text")]
                # Only try to infer heading from document_headings if we don't
                # already have an explicit "heading" on the chunk.
                if heading_text is None and char_start is not None:
                    heading_text, heading_path = _build_heading_path(headings, char_start)

            # Normalize a short document title for display/citation
            doc_title = None
            if file_path:
                try:
                    doc_title = Path(str(file_path)).name
                except Exception:
                    doc_title = str(file_path)

            # Build a human-readable citation string
            citation = _build_citation(
                metadata,
                heading_text=heading_text,
            )

            # Enrich metadata with heading/file information for the client
            if doc_title:
                metadata.setdefault("file_name", doc_title)
            if heading_text:
                metadata["heading"] = heading_text
            if heading_path:
                metadata["heading_path"] = heading_path

            # Annotate headings *inside* the chunk text so the model can see them clearly.
            if raw_text and heading_titles:
                annotated_lines: list[str] = []
                for line in raw_text.splitlines():
                    stripped = line.strip()
                    # Match heading titles case-insensitively after stripping
                    if stripped and stripped.lower() in {t.lower() for t in heading_titles}:
                        annotated_lines.append(f"=== HEADING: {stripped} ===")
                    else:
                        annotated_lines.append(line)
                chunk_text = "\n".join(annotated_lines)
            else:
                chunk_text = raw_text

            # Build a header block that is prepended to the chunk text so the LLM
            # can *see* and easily copy the exact source information (file + heading).
            header_lines: list[str] = []
            if doc_title:
                header_lines.append(f"SOURCE_FILE: {doc_title}")
            if heading_path:
                header_lines.append(f"SOURCE_HEADING_PATH: {' > '.join(heading_path)}")
            elif heading_text:
                header_lines.append(f"SOURCE_HEADING: {heading_text}")

            # Also surface the fully formatted citation string prominently so that
            # calling LLMs are more likely to copy it into their final answers.
            if citation:
                header_lines.append(f"SOURCE_CITATION: {citation}")
            # Fallback: if we have document-level headings but no position info,
            # still expose them so the model can see the section names.
            elif headings:
                normalized_headings = [h.get("text", "").strip() for h in headings if h.get("text")]
                if normalized_headings:
                    header_lines.append(f"ALL_DOCUMENT_HEADINGS: {' | '.join(normalized_headings)}")

            if header_lines:
                header_lines.append("")  # blank line
                header_lines.append("Content:")
                header_lines.append("")  # blank line
                header = "\n".join(header_lines)
                display_text = f"{header}{chunk_text.lstrip() if chunk_text else ''}"
            else:
                display_text = chunk_text

            location = {
                "file": file_path or "Unknown source",
                "page": metadata.get("page_label")
                or metadata.get("page_number")
                or metadata.get("page"),
                "heading": heading_text,
                "heading_path": heading_path or None,
            }
            chunk_data = {
                "text": display_text,  # Full content, not truncated, with header prefix
                "score": round(float(node.score), 3) if hasattr(node, 'score') and node.score is not None else 0.0,
                "metadata": metadata,
                "citation": citation,
                "location": location,
            }
            citation_list.append(citation)
            chunks.append(chunk_data)

        elapsed = time.time() - start_time

        # Deduplicate citations while preserving order
        unique_citations = list(dict.fromkeys(citation_list))

        return {
            "chunks": chunks,
            "query": query,
            "num_chunks": len(chunks),
            "citations": unique_citations,
            "retrieval_time": f"{elapsed:.2f}s",
        }
        
    except Exception as e:
        return {
            "chunks": [],
            "error": str(e),
            "query": query,
        }


if __name__ == "__main__":
    mcp.run()

