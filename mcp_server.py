#!/usr/bin/env python3
"""
MCP server for querying documentation using RAG.
Returns raw document chunks for the calling LLM to synthesize.
"""

import asyncio
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

from mcp.server.fastmcp import FastMCP

# Import from refactored modules
from opd_mcp.config import (
    STORAGE_DIR,
    RETRIEVAL_EMBED_MODEL_NAME,
    RETRIEVAL_RERANK_MODEL_NAME,
    RETRIEVAL_EMBED_TOP_K,
    RETRIEVAL_RERANK_TOP_K,
    RETRIEVAL_RECENCY_BOOST,
    RETRIEVAL_RECENCY_HALF_LIFE_DAYS,
    RETRIEVAL_SCORE_THRESHOLD,
    CONFLUENCE_TIMEOUT,
    CA_BUNDLE_PATH,
    DATA_DIR,
    configure_ssl,
    configure_offline_mode,
    is_offline_mode,
)
from opd_mcp.utils.logging import rotate_log_if_needed
from opd_mcp.utils.text import preprocess_query
from opd_mcp.models.reranking import ensure_reranker
from opd_mcp.storage.headings import get_heading_store
from opd_mcp.retrieval.index import load_llamaindex_index
from opd_mcp.retrieval.ranking import (
    reciprocal_rank_fusion,
    apply_recency_boost,
    filter_nodes_by_date,
)
from opd_mcp.retrieval.bm25 import ensure_bm25_retriever, expand_filename_matches
from opd_mcp.retrieval.confluence import search_confluence
from opd_mcp.retrieval.location import build_heading_path, char_offset_to_line

# Log file configuration
LOG_FILE = "mcp.log"
LOG_MAX_SIZE_MB = 10

# Rotate log file if needed before setting up logging
rotate_log_if_needed(LOG_FILE, LOG_MAX_SIZE_MB)

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)

# Configure SSL
configure_ssl(logger)

# Configure offline mode
_offline_mode = is_offline_mode()
if _offline_mode:
    configure_offline_mode(True)

# Initialize FastMCP server
mcp = FastMCP("llamaindex-docs-rag")

# Startup log buffer - populated at module load, flushed on first list_tools call
_startup_log_buffer: list[str] = []
_startup_logs_flushed = False


def _collect_startup_info():
    """Collect startup configuration info into the buffer."""
    global _startup_log_buffer

    _startup_log_buffer.append("MCP Server Startup")
    _startup_log_buffer.append("==================")
    _startup_log_buffer.append(
        f"Storage Directory: {STORAGE_DIR.resolve()} (Exists: {STORAGE_DIR.exists()})"
    )
    _startup_log_buffer.append(f"Embedding Model: {RETRIEVAL_EMBED_MODEL_NAME}")
    _startup_log_buffer.append(f"Rerank Model: {RETRIEVAL_RERANK_MODEL_NAME}")
    _startup_log_buffer.append(f"Offline Mode: {_offline_mode}")

    # Log CA bundle configuration
    if CA_BUNDLE_PATH:
        ca_bundle_path = Path(CA_BUNDLE_PATH)
        if ca_bundle_path.exists():
            _startup_log_buffer.append(f"CA Bundle: {ca_bundle_path.resolve()}")
        else:
            _startup_log_buffer.append(f"CA Bundle: {ca_bundle_path.resolve()} (NOT FOUND)")
    else:
        _startup_log_buffer.append("CA Bundle: Not configured (using system defaults)")

    # Log indexed document stats
    db_path = STORAGE_DIR / "ingestion_state.db"
    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("SELECT path, doc_ids FROM files")
                files = cursor.fetchall()
                file_count = len(files)
                total_chunks = sum(len(row[1].split(",")) if row[1] else 0 for row in files)
                _startup_log_buffer.append(
                    f"Indexed Documents: {file_count} files ({total_chunks} chunks)"
                )
        except Exception as e:
            _startup_log_buffer.append(f"Failed to read ingestion state: {e}")
    else:
        _startup_log_buffer.append("Indexed Documents: 0 (No ingestion state found)")

    # Log Confluence status
    confluence_url = os.getenv("CONFLUENCE_URL")
    if confluence_url:
        _startup_log_buffer.append(f"Confluence Integration: ENABLED (URL: {confluence_url})")
        try:
            from llama_index.readers.confluence import ConfluenceReader

            username = os.getenv("CONFLUENCE_USERNAME")
            api_token = os.getenv("CONFLUENCE_API_TOKEN")
            if username and api_token:
                ConfluenceReader(
                    base_url=confluence_url, user_name=username, password=api_token
                )
                _startup_log_buffer.append(
                    "Confluence Connection: Credentials provided, reader initialized."
                )
            else:
                _startup_log_buffer.append(
                    "Confluence Configuration: Missing USERNAME or API_TOKEN"
                )
        except ImportError:
            _startup_log_buffer.append(
                "Confluence Integration: Skipped (Library not installed)"
            )
        except Exception as e:
            _startup_log_buffer.append(f"Confluence Connection Failed: {e}")
    else:
        _startup_log_buffer.append("Confluence Integration: DISABLED (CONFLUENCE_URL not set)")


# Collect startup info at module load
_collect_startup_info()


# Wrap list_tools to flush startup logs on first client connection
_original_list_tools = mcp.list_tools


async def _wrapped_list_tools():
    """Wrapper that flushes startup logs on first list_tools call."""
    global _startup_logs_flushed

    if not _startup_logs_flushed and _startup_log_buffer:
        _startup_logs_flushed = True
        try:
            ctx = mcp.get_context()
            for msg in _startup_log_buffer:
                await ctx.info(msg)
        except Exception as e:
            logger.warning(f"Could not flush startup logs via MCP: {e}")
            for msg in _startup_log_buffer:
                logger.info(msg)

    return await _original_list_tools()


# Replace the list_tools handler
mcp._mcp_server.list_tools()(_wrapped_list_tools)


@mcp.tool()
async def retrieve_docs(
    query: str,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """
Search the local documentation corpus and return the most relevant chunks.

Args:
    query: Search query text
    date_from: Optional start date filter (YYYY-MM-DD format, inclusive)
    date_to: Optional end date filter (YYYY-MM-DD format, inclusive)

The tool returns a structured response with:
- `chunks`: Array of retrieved document chunks
- `query`: The search query used
- `num_chunks`: Number of chunks returned
- `retrieval_time`: Time taken for retrieval
- `sources`: Array of unique source documents with URI, name, and MIME type (displayed as clickable links in MCP clients)

Each chunk includes:
- `text`: Full chunk text content
- `location`: Structured location details (file, page, heading, heading_path)
- `metadata`: Original document metadata (includes creation_date and last_modified_date when available)
- `score`: Relevance score
    """
    start_time = time.time()

    try:
        # Preprocess query to improve retrieval quality
        enhanced_query = preprocess_query(query)

        # Load index
        index = load_llamaindex_index()

        # Stage 1a: Vector search (embedding-based semantic search)
        retriever = index.as_retriever(similarity_top_k=RETRIEVAL_EMBED_TOP_K)
        vector_nodes = retriever.retrieve(enhanced_query)

        # Stage 1b: BM25 file name search (keyword-based)
        bm25_retriever = ensure_bm25_retriever()
        if bm25_retriever:
            try:
                bm25_matches = bm25_retriever.retrieve(enhanced_query)
                if bm25_matches:
                    bm25_nodes = expand_filename_matches(bm25_matches, index.docstore)
                    logger.info(
                        f"BM25 matched {len(bm25_matches)} files, expanded to {len(bm25_nodes)} chunks"
                    )
                    nodes = reciprocal_rank_fusion([vector_nodes, bm25_nodes])
                    logger.info(
                        f"Fused {len(vector_nodes)} vector + {len(bm25_nodes)} BM25 -> {len(nodes)} unique nodes"
                    )
                else:
                    nodes = vector_nodes
            except Exception as e:
                logger.error(f"BM25 search failed, using vector-only results: {e}")
                nodes = vector_nodes
        else:
            nodes = vector_nodes

        # Search Confluence in parallel (with timeout)
        confluence_nodes = []
        if os.getenv("CONFLUENCE_URL"):
            try:
                loop = asyncio.get_running_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, search_confluence, enhanced_query),
                    timeout=CONFLUENCE_TIMEOUT,
                )
                confluence_nodes = result
                logger.info(f"Confluence search returned {len(confluence_nodes)} entries")
            except asyncio.TimeoutError:
                msg = f"Confluence search timed out after {CONFLUENCE_TIMEOUT}s"
                try:
                    ctx = mcp.get_context()
                    await ctx.warning(msg)
                except Exception:
                    logger.warning(msg)
            except Exception as e:
                msg = f"Error during Confluence search: {e}"
                try:
                    ctx = mcp.get_context()
                    await ctx.error(msg)
                except Exception:
                    logger.error(msg)

        if confluence_nodes:
            nodes.extend(confluence_nodes)

        # Apply date filtering if requested
        if date_from or date_to:
            original_count = len(nodes)
            nodes = filter_nodes_by_date(nodes, date_from, date_to)
            logger.info(f"Date filtering: {original_count} -> {len(nodes)} nodes")

        # Apply recency boost if configured
        if RETRIEVAL_RECENCY_BOOST > 0:
            nodes = apply_recency_boost(
                nodes, RETRIEVAL_RECENCY_BOOST, RETRIEVAL_RECENCY_HALF_LIFE_DAYS
            )

        # Rerank the retrieved nodes with FlashRank
        rerank_scores: dict[int, float] = {}
        if nodes:
            # Limit how many nodes to rerank (for performance) - rerank more than we return
            # to give the reranker enough candidates to find the best results
            rerank_input_limit = min(50, len(nodes))
            output_limit = max(1, min(RETRIEVAL_RERANK_TOP_K, len(nodes)))
            try:
                reranker = ensure_reranker()
                # Limit passages sent to reranker to avoid timeout on large result sets
                nodes_to_rerank = nodes[:rerank_input_limit]
                passages = [{"text": node.node.get_content() or ""} for node in nodes_to_rerank]

                from flashrank import RerankRequest

                rerank_request = RerankRequest(query=enhanced_query, passages=passages)
                reranked_results = reranker.rerank(rerank_request)

                # Create a mapping from document text to (index, node)
                text_to_indices = {}
                for idx, node in enumerate(nodes_to_rerank):
                    node_text = node.node.get_content() or ""
                    if node_text not in text_to_indices:
                        text_to_indices[node_text] = []
                    text_to_indices[node_text].append((idx, node))

                # Reorder nodes based on reranking results
                reranked_nodes = []
                seen_indices = set()
                for result in reranked_results:
                    doc_text = result.get("text", "")
                    score = result.get("score", 0.0)

                    if doc_text in text_to_indices:
                        for idx, node in text_to_indices[doc_text]:
                            if idx not in seen_indices:
                                reranked_nodes.append(node)
                                rerank_scores[id(node)] = float(score)
                                seen_indices.add(idx)
                                break

                # Add any nodes that weren't in reranked results
                for idx, node in enumerate(nodes_to_rerank):
                    if idx not in seen_indices:
                        reranked_nodes.append(node)

                nodes = reranked_nodes[:output_limit]
            except Exception as e:
                logger.error(f"Reranking failed, falling back to vector search order: {e}")
                nodes = nodes[:output_limit]

        # Filter nodes by score threshold
        if RETRIEVAL_SCORE_THRESHOLD > 0:
            nodes = [
                node
                for node in nodes
                if rerank_scores.get(id(node), 0.0) >= RETRIEVAL_SCORE_THRESHOLD
            ]

        # Format chunks with simplified structure
        chunks = []
        for node in nodes:
            metadata = dict(node.node.metadata or {})
            chunk_text = node.node.get_content()

            file_path = (
                metadata.get("file_path")
                or metadata.get("file_name")
                or metadata.get("source")
            )
            original_source = metadata.get("source")

            # Build heading path from document structure
            headings = metadata.get("document_headings") or metadata.get("headings") or []
            if not headings and file_path:
                headings = get_heading_store().get_headings(str(file_path))
            char_start = getattr(node.node, "start_char_idx", None)
            heading_text = metadata.get("heading")
            heading_path: list[str] = []
            if isinstance(headings, list) and headings:
                if heading_text is None and char_start is not None:
                    heading_text, heading_path = build_heading_path(headings, char_start)
            meta_heading_path = metadata.get("heading_path")
            if not heading_path and meta_heading_path:
                heading_path = list(meta_heading_path)
            if heading_text and (not heading_path or heading_path[-1] != heading_text):
                heading_path = (
                    heading_path + [heading_text] if heading_path else [heading_text]
                )

            # Build URI for the source document
            source_uri = None
            if original_source == "Confluence":
                confluence_url = os.getenv("CONFLUENCE_URL", "")
                page_id = metadata.get("page_id")
                if confluence_url and page_id:
                    source_uri = (
                        f"{confluence_url.rstrip('/')}/pages/viewpage.action?pageId={page_id}"
                    )
                elif confluence_url:
                    title = metadata.get("title", metadata.get("file_name", ""))
                    if title:
                        encoded_title = quote(title.replace(" ", "+"))
                        source_uri = f"{confluence_url.rstrip('/')}/spaces/~{encoded_title}"
            elif file_path:
                try:
                    file_path_obj = Path(str(file_path))
                    if file_path_obj.is_absolute():
                        source_uri = f"file://{file_path_obj.resolve()}"
                    else:
                        resolved_path = (DATA_DIR / file_path_obj).resolve()
                        source_uri = f"file://{resolved_path}"
                except Exception:
                    source_uri = None

            # Get page number (for PDFs and DOCX)
            page_number = (
                metadata.get("page_label")
                or metadata.get("page_number")
                or metadata.get("page")
            )

            # Get line number (for markdown/txt files)
            line_number = None
            line_offsets = metadata.get("line_offsets")
            if line_offsets and char_start is not None:
                line_number = char_offset_to_line(char_start, line_offsets)

            # Build simplified location object
            location = {
                "uri": source_uri,
                "file": str(file_path) if file_path else None,
                "page": page_number,
                "line": line_number,
                "heading_path": heading_path if heading_path else None,
            }

            # Build chunk with simplified structure
            score_value = rerank_scores.get(id(node), getattr(node, "score", None))
            chunk_data = {
                "text": chunk_text,
                "score": round(float(score_value), 3) if score_value is not None else 0.0,
                "location": location,
            }
            chunks.append(chunk_data)

        elapsed = time.time() - start_time

        structured_response = {
            "chunks": chunks,
            "query": query,
            "num_chunks": len(chunks),
            "retrieval_time": f"{elapsed:.2f}s",
        }

        return structured_response

    except Exception as e:
        logger.error(f"Error in retrieve_docs: {e}", exc_info=True)
        error_response = {
            "chunks": [],
            "error": str(e),
            "query": query,
        }
        return error_response


if __name__ == "__main__":
    mcp.run()
