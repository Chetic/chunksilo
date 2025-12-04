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
from llama_index.embeddings.fastembed import FastEmbedEmbedding

# Load environment variables
load_dotenv()

# Configuration
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "./storage"))

# Two-stage retrieval configuration
# Stage 1: vector search over embeddings
RETRIEVAL_EMBED_TOP_K = int(os.getenv("RETRIEVAL_EMBED_TOP_K", "10"))
RETRIEVAL_EMBED_MODEL_NAME = os.getenv(
    "RETRIEVAL_EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5"
)

# Stage 2: FlashRank reranking (CPU-only, ONNX-based) of the vector search candidates
RETRIEVAL_RERANK_TOP_K = int(os.getenv("RETRIEVAL_RERANK_TOP_K", "5"))
RETRIEVAL_RERANK_MODEL_NAME = os.getenv(
    "RETRIEVAL_RERANK_MODEL_NAME", "ms-marco-MiniLM-L-12-v2"
)

# Shared cache directory for embedding and reranker models
RETRIEVAL_MODEL_CACHE_DIR = Path(os.getenv("RETRIEVAL_MODEL_CACHE_DIR", "./models"))

# Configure offline mode for HuggingFace libraries to prevent network requests
# The MCP server is intended to run in offline environments where models are already cached.
# Set OFFLINE=0 in environment to allow network access if needed.
_offline_mode = os.getenv("OFFLINE", "1").lower() not in ("0", "false", "no")
if _offline_mode:
    # Enable offline mode to prevent HuggingFace libraries from making network requests
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    # Point HuggingFace Hub to our cache directory so it can find cached models
    cache_dir_abs = RETRIEVAL_MODEL_CACHE_DIR.resolve()
    os.environ["HF_HOME"] = str(cache_dir_abs)
    os.environ["HF_HUB_CACHE"] = str(cache_dir_abs)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir_abs)

# Initialize FastMCP server
mcp = FastMCP("llamaindex-docs-rag")

# Global caches
_index_cache = None
_embed_model_initialized = False
_reranker_model = None


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


def _get_cached_model_path(cache_dir: Path, model_name: str) -> Path | None:
    """
    Get the cached model directory path using huggingface_hub's snapshot_download.
    This works completely offline and bypasses fastembed's download_model API calls.
    
    Args:
        cache_dir: Directory where models are cached
        model_name: FastEmbed model name (e.g., 'BAAI/bge-small-en-v1.5')
        
    Returns:
        Path to the cached model directory, or None if not found or huggingface_hub unavailable
    """
    try:
        from huggingface_hub import snapshot_download
        from fastembed import TextEmbedding
        models = TextEmbedding.list_supported_models()
        model_info = [m for m in models if m.get("model") == model_name]
        if model_info:
            hf_source = model_info[0].get("sources", {}).get("hf")
            if hf_source:
                cache_dir_abs = cache_dir.resolve()
                # Use snapshot_download with local_files_only=True to get cached model path
                # This works completely offline and bypasses fastembed's API call
                model_dir = snapshot_download(
                    repo_id=hf_source,
                    local_files_only=True,
                    cache_dir=str(cache_dir_abs)
                )
                return Path(model_dir).resolve()
    except (ImportError, Exception):
        # huggingface_hub might not be available, or model not in cache
        pass
    return None


def _ensure_embed_model():
    """
    Ensure the same embedding model used during ingestion is available at query time.

    If this is not set, LlamaIndex falls back to its default (typically an OpenAI
    embedding model), which would require an OPENAI_API_KEY and cause failures
    inside the MCP server.
    
    Uses cached model path in offline mode to bypass fastembed's download_model API calls.
    """
    global _embed_model_initialized

    if _embed_model_initialized:
        return

    # Try to use cached model path in offline mode to bypass fastembed's download step
    cached_model_path = _get_cached_model_path(
        RETRIEVAL_MODEL_CACHE_DIR, RETRIEVAL_EMBED_MODEL_NAME
    )
    if cached_model_path and _offline_mode:
        # Use specific_model_path to bypass fastembed's download_model API call
        embed_model = FastEmbedEmbedding(
            model_name=RETRIEVAL_EMBED_MODEL_NAME,
            cache_dir=str(RETRIEVAL_MODEL_CACHE_DIR),
            specific_model_path=str(cached_model_path)
        )
    else:
        # Normal initialization (non-offline or cached path not available)
        embed_model = FastEmbedEmbedding(
            model_name=RETRIEVAL_EMBED_MODEL_NAME,
            cache_dir=str(RETRIEVAL_MODEL_CACHE_DIR),
        )
    Settings.embed_model = embed_model
    _embed_model_initialized = True


def _ensure_reranker():
    """Load the FlashRank reranking model for CPU inference."""

    global _reranker_model

    if _reranker_model is not None:
        return _reranker_model

    try:
        from flashrank import Ranker
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "flashrank is required for reranking. Please install dependencies from requirements.txt."
        ) from exc

    # FlashRank uses ONNX models and handles caching internally
    # The model name should be a FlashRank-compatible model identifier
    # Default to a lightweight model if the configured one isn't FlashRank-compatible
    model_name = RETRIEVAL_RERANK_MODEL_NAME
    
    # Map cross-encoder model names to FlashRank equivalents if needed
    # FlashRank supports models like 'ms-marco-TinyBERT-L-2-v2', 'ms-marco-MiniLM-L-12-v2', etc.
    # Note: FlashRank doesn't have L-6 models, so we map to L-12 equivalents
    model_mapping = {
        "cross-encoder/ms-marco-MiniLM-L-6-v2": "ms-marco-MiniLM-L-12-v2",  # L-6 not available, use L-12
        "ms-marco-MiniLM-L-6-v2": "ms-marco-MiniLM-L-12-v2",  # Direct mapping for L-6
    }
    if model_name in model_mapping:
        model_name = model_mapping[model_name]
    elif model_name.startswith("cross-encoder/"):
        # Extract model name after cross-encoder/ prefix and try to map
        base_name = model_name.replace("cross-encoder/", "")
        # If it's an L-6 model, map to L-12
        if "L-6" in base_name:
            model_name = base_name.replace("L-6", "L-12")
        else:
            model_name = base_name

    try:
        _reranker_model = Ranker(model_name=model_name, cache_dir=str(RETRIEVAL_MODEL_CACHE_DIR))
    except Exception as exc:
        if _offline_mode:
            raise FileNotFoundError(
                f"Rerank model '{model_name}' not available in cache directory {RETRIEVAL_MODEL_CACHE_DIR}. "
                "Download it before running in offline mode."
            ) from exc
        raise
    
    return _reranker_model


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
    # query embeddings use the same model as ingestion (FastEmbed, not OpenAI).
    _ensure_embed_model()

    storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    index = load_index_from_storage(storage_context)
    _index_cache = index
    return index


@mcp.tool()
async def retrieve_docs(query: str) -> dict[str, Any]:
    """
Search the local PDF / DOCX / Markdown documentation corpus and return the most relevant chunks.

Your primary responsibility when using this tool is not just to find information, but to **show clearly where it came from.** That means:

1. **Every time you use information from a retrieved chunk in your answer, you must add a source citation.**
2. **Always place each citation on its own separate line, immediately after the sentence or paragraph it supports.**
3. **An answer without citations is considered incomplete and may be treated as incorrect, even if the content is otherwise good.**

When this tool returns results, each chunk includes:

- `text`: Full chunk text content (may start with `SOURCE_*` helper lines).
- `citation`: A human-readable citation string. **This is what you should paste or adapt directly into your answer when you reference that chunk, on its own line.**
- `location`: Structured location details (file, page, heading, heading_path).
- `metadata`: Original document metadata.

At the top level, the tool response also includes a `citations` array listing all unique citation strings.

To write a high-quality answer:

- As you draft your reasoning or explanation, **immediately attach the appropriate `citation` string whenever you use a fact, definition, procedure, or example from a chunk.**
- **Put that citation on a new line by itself**, directly after the relevant text. For example:

  - Your explanatory text here…
  - `Citation: <paste citation string here>`

- Prefer citing **exactly the chunks you actually used**, not the whole list of returned results.

Before you consider your answer finished, do a quick check:

- “Have I added at least one citation for every distinct document I used?”
- “Are all my citations on their own separate lines, immediately after the text they support?”

Following these steps is **mandatory** for a complete, trustworthy response.
    """
    start_time = time.time()
    
    try:
        # Load index
        index = load_llamaindex_index()

        # Use retriever (no LLM needed - just retrieval)
        retriever = index.as_retriever(similarity_top_k=RETRIEVAL_EMBED_TOP_K)

        # Retrieve relevant chunks (embedding stage)
        nodes = retriever.retrieve(query)

        # Rerank the retrieved nodes with FlashRank (CPU-only, ONNX-based) and trim to the
        # configured final Top K for the tool response.
        rerank_scores: dict[int, float] = {}
        if nodes:
            rerank_limit = max(1, min(RETRIEVAL_RERANK_TOP_K, len(nodes)))
            try:
                reranker = _ensure_reranker()
                # Prepare documents for reranking
                documents = [node.node.get_content() or "" for node in nodes]
                # FlashRank returns reranked documents (list of dicts with 'text' and 'score')
                reranked_results = reranker.rerank(query, documents)
                
                # Create a mapping from document text to (index, node) for reliable matching
                # Use index as primary key to handle duplicate text correctly
                text_to_indices = {}
                for idx, node in enumerate(nodes):
                    node_text = node.node.get_content() or ""
                    if node_text not in text_to_indices:
                        text_to_indices[node_text] = []
                    text_to_indices[node_text].append((idx, node))
                
                # Reorder nodes based on reranking results
                reranked_nodes = []
                seen_indices = set()
                for result in reranked_results:
                    # Handle both dict format {'text': ..., 'score': ...} and string format
                    if isinstance(result, dict):
                        doc_text = result.get("text", "")
                        score = result.get("score", 0.0)
                    else:
                        # If result is just a string, use it as text
                        doc_text = str(result)
                        score = 0.0
                    
                    if doc_text in text_to_indices:
                        # Get the first unused (index, node) pair for this text
                        for idx, node in text_to_indices[doc_text]:
                            if idx not in seen_indices:
                                reranked_nodes.append(node)
                                rerank_scores[id(node)] = float(score)
                                seen_indices.add(idx)
                                break
                
                # Add any nodes that weren't in reranked results (shouldn't happen, but safety)
                for idx, node in enumerate(nodes):
                    if idx not in seen_indices:
                        reranked_nodes.append(node)
                
                nodes = reranked_nodes[:rerank_limit]
            except Exception:
                # Fall back to vector search ordering if reranking fails
                nodes = nodes[:rerank_limit]

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
            score_value = rerank_scores.get(id(node), getattr(node, "score", None))
            chunk_data = {
                "text": display_text,  # Full content, not truncated, with header prefix
                "score": round(float(score_value), 3) if score_value is not None else 0.0,
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

