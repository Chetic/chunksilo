#!/usr/bin/env python3
"""
MCP server for querying documentation using RAG.
Returns raw document chunks for the calling LLM to synthesize.
"""
import os
import sys
import time
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP, Context
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
try:
    from llama_index.readers.confluence import ConfluenceReader
    import requests  # Available when llama-index-readers-confluence is installed
except ImportError:
    ConfluenceReader = None
    requests = None

import asyncio
import math
import logging

# Log file configuration
LOG_FILE = "mcp.log"
LOG_MAX_SIZE_MB = 10
LOG_MAX_SIZE_BYTES = LOG_MAX_SIZE_MB * 1024 * 1024  # 10MB in bytes


def _rotate_log_if_needed():
    """Rotate log file if it exists and is over the size limit."""
    log_path = Path(LOG_FILE)
    if log_path.exists() and log_path.stat().st_size > LOG_MAX_SIZE_BYTES:
        # Generate timestamp suffix with microsecond precision and process ID
        # to prevent collisions when multiple processes rotate simultaneously
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        process_id = os.getpid()
        rotated_name = f"mcp_{timestamp}_{process_id}.log"
        rotated_path = log_path.parent / rotated_name
        log_path.rename(rotated_path)
        # Create new empty log file
        log_path.touch()


# Rotate log file if needed before setting up logging
_rotate_log_if_needed()

# Set up logging
# We primarily use MCP logging context, but keep a stderr logger for unhandled/startup issues
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),  # file
        logging.StreamHandler(sys.stderr),                 # stderr
    ],
)

# Load environment variables

# Configuration
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "./storage"))

# Two-stage retrieval configuration
# Stage 1: vector search over embeddings
# Higher values provide more candidates for reranking, improving precision
RETRIEVAL_EMBED_TOP_K = int(os.getenv("RETRIEVAL_EMBED_TOP_K", "20"))
RETRIEVAL_EMBED_MODEL_NAME = os.getenv(
    "RETRIEVAL_EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5"
)

# Stage 2: FlashRank reranking (CPU-only, ONNX-based) of the vector search candidates
RETRIEVAL_RERANK_TOP_K = int(os.getenv("RETRIEVAL_RERANK_TOP_K", "5"))
RETRIEVAL_RERANK_MODEL_NAME = os.getenv(
    "RETRIEVAL_RERANK_MODEL_NAME", "ms-marco-MiniLM-L-12-v2"
)

# Shared cache directory for embedding and reranking models
RETRIEVAL_MODEL_CACHE_DIR = Path(os.getenv("RETRIEVAL_MODEL_CACHE_DIR", "./models"))

# Confluence Configuration
CONFLUENCE_TIMEOUT = float(os.getenv("CONFLUENCE_TIMEOUT", "10.0"))
CONFLUENCE_MAX_RESULTS = int(os.getenv("CONFLUENCE_MAX_RESULTS", "30"))

# Recency boost configuration
# Documents are boosted based on their age using exponential decay
RETRIEVAL_RECENCY_BOOST = float(os.getenv("RETRIEVAL_RECENCY_BOOST", "0.3"))  # 0.0-1.0
RETRIEVAL_RECENCY_HALF_LIFE_DAYS = int(os.getenv("RETRIEVAL_RECENCY_HALF_LIFE_DAYS", "365"))

# Score threshold for filtering low-relevance results
RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.1"))

# Common English stopwords to filter from Confluence CQL queries
# These words add no search value and can cause overly broad/narrow results
CONFLUENCE_STOPWORDS = frozenset({
    # Articles
    "a", "an", "the",
    # Prepositions
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    # Conjunctions
    "and", "or", "but", "if", "then", "so",
    # Pronouns
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "she", "it", "they", "them",
    # Common verbs
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "can", "could", "will", "would", "should", "may", "might", "must",
    # Question words
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    # Other common words
    "this", "that", "these", "those", "here", "there", "all", "any", "each", "some", "no", "not",
    "about", "into", "over", "after", "before", "between", "under", "again", "just", "only", "also",
})

# SSL/TLS Configuration
# CA bundle path for HTTPS connections (e.g., Confluence, future HTTP clients)
# If set, this will be used by requests, urllib3, and other HTTP libraries
CA_BUNDLE_PATH = os.getenv("CA_BUNDLE_PATH")
if CA_BUNDLE_PATH:
    ca_bundle_path = Path(CA_BUNDLE_PATH)
    if ca_bundle_path.exists():
        # Set environment variables that requests, urllib3, and other libraries respect
        os.environ["REQUESTS_CA_BUNDLE"] = str(ca_bundle_path.resolve())
        os.environ["SSL_CERT_FILE"] = str(ca_bundle_path.resolve())
        logger.info(f"CA bundle configured: {ca_bundle_path.resolve()}")
    else:
        logger.warning(f"CA bundle path does not exist: {ca_bundle_path.resolve()}")

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


# Startup log buffer - populated at module load, flushed on first list_tools call
_startup_log_buffer: list[str] = []
_startup_logs_flushed = False


def _collect_startup_info():
    """Collect startup configuration info into the buffer."""
    global _startup_log_buffer
    _startup_log_buffer.append("MCP Server Startup")
    _startup_log_buffer.append("==================")
    _startup_log_buffer.append(f"Storage Directory: {STORAGE_DIR.resolve()} (Exists: {STORAGE_DIR.exists()})")
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
                _startup_log_buffer.append(f"Indexed Documents: {file_count} files ({total_chunks} chunks)")
        except Exception as e:
            _startup_log_buffer.append(f"Failed to read ingestion state: {e}")
    else:
        _startup_log_buffer.append("Indexed Documents: 0 (No ingestion state found)")

    # Log Confluence status
    confluence_url = os.getenv("CONFLUENCE_URL")
    if confluence_url:
        _startup_log_buffer.append(f"Confluence Integration: ENABLED (URL: {confluence_url})")
        if ConfluenceReader:
            try:
                username = os.getenv("CONFLUENCE_USERNAME")
                api_token = os.getenv("CONFLUENCE_API_TOKEN")
                if username and api_token:
                    reader = ConfluenceReader(base_url=confluence_url, user_name=username, password=api_token)
                    _startup_log_buffer.append("Confluence Connection: Credentials provided, reader initialized.")
                else:
                    _startup_log_buffer.append("Confluence Configuration: Missing USERNAME or API_TOKEN")
            except Exception as e:
                _startup_log_buffer.append(f"Confluence Connection Failed: {e}")
        else:
            _startup_log_buffer.append("Confluence Integration: Skipped (Library not installed)")
    else:
        _startup_log_buffer.append("Confluence Integration: DISABLED (CONFLUENCE_URL not set)")


# Collect startup info at module load
_collect_startup_info()


# Wrap list_tools to flush startup logs on first client connection
_original_list_tools = mcp.list_tools


async def _wrapped_list_tools():
    """Wrapper that flushes startup logs on first list_tools call."""
    global _startup_logs_flushed
    
    # Flush buffered startup logs on first call
    if not _startup_logs_flushed and _startup_log_buffer:
        _startup_logs_flushed = True
        try:
            ctx = mcp.get_context()
            for msg in _startup_log_buffer:
                await ctx.info(msg)
        except Exception as e:
            # Fallback to stderr if context not available
            logger.warning(f"Could not flush startup logs via MCP: {e}")
            for msg in _startup_log_buffer:
                logger.info(msg)
    
    # Call original list_tools
    return await _original_list_tools()


# Replace the list_tools handler
mcp._mcp_server.list_tools()(_wrapped_list_tools)


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


def _char_offset_to_line(char_offset: int | None, line_offsets: list[int] | None) -> int | None:
    """
    Convert a character offset to a line number (1-indexed) using precomputed line offsets.

    Args:
        char_offset: Character position in the document (0-indexed)
        line_offsets: List where line_offsets[i] is the char position where line i+1 starts

    Returns:
        Line number (1-indexed) or None if cannot be determined
    """
    if char_offset is None or not line_offsets:
        return None

    # Binary search to find the line containing char_offset
    left, right = 0, len(line_offsets) - 1
    while left < right:
        mid = (left + right + 1) // 2
        if line_offsets[mid] <= char_offset:
            left = mid
        else:
            right = mid - 1

    return left + 1  # Convert to 1-indexed line number


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
        logger.info(f"Loading embedding model from cache: {cached_model_path}")
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
    logger.info("Embedding model initialized successfully")
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
    
    logger.info(f"Rerank model '{model_name}' loaded successfully")
    return _reranker_model


def _preprocess_query(query: str) -> str:
    """
    Preprocess queries with basic normalization.
    
    Techniques applied:
    - Normalize whitespace
    - Remove trailing punctuation that might interfere with matching
    
    Returns the original query if preprocessing results in an empty string.
    """
    if not query or not query.strip():
        return query
    
    # Store original query to preserve it if preprocessing results in empty string
    original_query = query
    
    # Normalize whitespace (collapse multiple spaces)
    query = " ".join(query.split())
    
    # Remove trailing punctuation that might interfere with matching
    query = query.rstrip(".,!?;")
    
    # If preprocessing resulted in an empty string, return original query
    processed = query.strip()
    return processed if processed else original_query


def _prepare_confluence_query_terms(query: str) -> list[str]:
    """
    Prepare query terms for Confluence CQL search.

    Processing steps:
    1. Split query into words and lowercase
    2. Filter out stopwords
    3. Filter out very short words (< 2 chars)
    4. Escape special characters

    Args:
        query: The raw search query string

    Returns:
        List of prepared search terms (may be empty if all words are stopwords)
    """
    words = query.strip().lower().split()
    meaningful = [w for w in words if w not in CONFLUENCE_STOPWORDS and len(w) >= 2]
    return [w.replace('"', '\\"') for w in meaningful]


def _get_confluence_page_dates(
    base_url: str, page_id: str, username: str, api_token: str
) -> dict[str, str]:
    """
    Fetch creation and modification dates for a Confluence page.

    Args:
        base_url: Confluence base URL
        page_id: The Confluence page ID
        username: Confluence username
        api_token: Confluence API token

    Returns:
        Dict with 'creation_date' and/or 'last_modified_date' in YYYY-MM-DD format
    """
    if requests is None:
        return {}

    try:
        # Use v2 API to get page with version info
        url = f"{base_url.rstrip('/')}/wiki/api/v2/pages/{page_id}"
        response = requests.get(
            url,
            auth=(username, api_token),
            timeout=5.0,
            verify=CA_BUNDLE_PATH if CA_BUNDLE_PATH else True,
        )
        if response.status_code == 200:
            data = response.json()
            result = {}
            if "createdAt" in data:
                # Parse ISO format: "2023-01-25T09:40:17.506Z"
                try:
                    dt = datetime.fromisoformat(data["createdAt"].replace("Z", "+00:00"))
                    result["creation_date"] = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
            if "version" in data and "createdAt" in data["version"]:
                try:
                    dt = datetime.fromisoformat(data["version"]["createdAt"].replace("Z", "+00:00"))
                    result["last_modified_date"] = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
            return result
    except Exception as e:
        logger.debug(f"Failed to fetch Confluence page dates for {page_id}: {e}")
    return {}


def _search_confluence(query: str) -> list[NodeWithScore]:
    """
    Search Confluence for documents matching the query using CQL.
    Returns a list of NodeWithScore objects compatible with the reranker.

    Uses OR logic for multi-word queries to cast a wider net, relying on the
    FlashRank reranker to identify the most semantically relevant results.
    Filters out common stopwords to improve search precision.
    """
    base_url = os.getenv("CONFLUENCE_URL")
    # If CONFLUENCE_URL is unset or empty, disable search completely
    if not base_url:
        # Keep this as a warning because it can explain "0 entries" quickly.
        logger.warning("Confluence search skipped: CONFLUENCE_URL not set")
        return []

    if ConfluenceReader is None:
        logger.warning("llama-index-readers-confluence not installed, skipping Confluence search")
        return []

    username = os.getenv("CONFLUENCE_USERNAME")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")

    if not (base_url and username and api_token):
        missing = []
        if not username:
            missing.append("CONFLUENCE_USERNAME")
        if not api_token:
            missing.append("CONFLUENCE_API_TOKEN")
        logger.warning(f"Confluence search skipped: missing {', '.join(missing)}")
        return []

    try:
        reader = ConfluenceReader(base_url=base_url, user_name=username, password=api_token)

        # Prepare query terms (filter stopwords, escape special chars)
        query_terms = _prepare_confluence_query_terms(query)

        # Build CQL query using OR logic to cast a wider net
        if not query_terms:
            # All words were stopwords - fall back to using original query as phrase
            escaped = query.strip().replace('"', '\\"')
            if not escaped:
                logger.warning("Confluence search skipped: empty query after processing")
                return []
            cql = f'text ~ "{escaped}" AND type = "page"'
        elif len(query_terms) == 1:
            # Single meaningful word
            cql = f'text ~ "{query_terms[0]}" AND type = "page"'
        else:
            # Multiple words: use OR logic to find pages with ANY matching word
            # The reranker will sort by semantic relevance
            text_conditions = ' OR '.join([f'text ~ "{term}"' for term in query_terms])
            cql = f'({text_conditions}) AND type = "page"'

        logger.debug(f"Confluence CQL query: {cql}")
        documents = reader.load_data(cql=cql, max_results=CONFLUENCE_MAX_RESULTS)

        nodes: list[NodeWithScore] = []
        for doc in documents:
            # Create a TextNode from the Confluence document
            # Map standard Confluence metadata to our format
            metadata = doc.metadata.copy()
            metadata["source"] = "Confluence"
            if "title" in metadata:
                 metadata["file_name"] = metadata["title"]  # Use title as filename

            # Fetch dates for this page
            page_id = metadata.get("page_id")
            if page_id:
                date_info = _get_confluence_page_dates(base_url, page_id, username, api_token)
                metadata.update(date_info)

            node = TextNode(text=doc.text, metadata=metadata)
            # Assign a default score of 0.0 (will be updated by reranker)
            nodes.append(NodeWithScore(node=node, score=0.0))

        return nodes

    except Exception as e:
        logger.error(f"Failed to search Confluence: {e}", exc_info=True)
        return []


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
    
    logger.info("Loading LlamaIndex from storage...")

    # Make sure the embedding model is configured before using the index so that
    # query embeddings use the same model as ingestion (FastEmbed, not OpenAI).
    _ensure_embed_model()

    storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    index = load_index_from_storage(storage_context)
    _index_cache = index
    return index


def _parse_date(date_str: str) -> datetime | None:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _filter_nodes_by_date(
    nodes: list[NodeWithScore],
    date_from: str | None,
    date_to: str | None
) -> list[NodeWithScore]:
    """
    Filter nodes by date range.

    Args:
        nodes: List of nodes to filter
        date_from: Optional start date (YYYY-MM-DD, inclusive)
        date_to: Optional end date (YYYY-MM-DD, inclusive)

    Returns:
        Filtered list of nodes. Nodes without dates pass through for backward compatibility.
    """
    if not date_from and not date_to:
        return nodes

    from_dt = _parse_date(date_from) if date_from else None
    to_dt = _parse_date(date_to) if date_to else None

    filtered = []
    for node in nodes:
        metadata = node.node.metadata or {}
        # Check last_modified_date first, fall back to creation_date
        doc_date_str = metadata.get("last_modified_date") or metadata.get("creation_date")
        if not doc_date_str:
            # No date info - include by default (backward compatibility)
            filtered.append(node)
            continue

        doc_date = _parse_date(doc_date_str)
        if not doc_date:
            filtered.append(node)
            continue

        # Apply filters
        if from_dt and doc_date < from_dt:
            continue
        if to_dt and doc_date > to_dt:
            continue

        filtered.append(node)

    return filtered


def _apply_recency_boost(
    nodes: list[NodeWithScore],
    boost_weight: float,
    half_life_days: int = 365
) -> list[NodeWithScore]:
    """
    Apply time-decay boost to nodes based on document recency.

    Args:
        nodes: List of nodes to boost
        boost_weight: How much to weight recency (0.0 = no boost, 1.0 = recency dominates)
        half_life_days: Days until a document's recency boost is halved

    Returns:
        Nodes with adjusted scores, re-sorted by boosted score
    """
    if not nodes or boost_weight <= 0:
        return nodes

    today = datetime.now()
    boosted_nodes = []

    for node in nodes:
        metadata = node.node.metadata or {}
        doc_date_str = metadata.get("last_modified_date") or metadata.get("creation_date")

        # Calculate base score (or default to 0.5 if no score)
        base_score = node.score if node.score is not None else 0.5

        if not doc_date_str:
            # No date - use base score only
            boosted_nodes.append(NodeWithScore(node=node.node, score=base_score))
            continue

        doc_date = _parse_date(doc_date_str)
        if not doc_date:
            boosted_nodes.append(NodeWithScore(node=node.node, score=base_score))
            continue

        # Calculate age in days
        age_days = (today - doc_date).days
        if age_days < 0:
            age_days = 0  # Future dates treated as today

        # Exponential decay: recency_factor = 0.5^(age/half_life)
        decay_rate = math.log(2) / half_life_days
        recency_factor = math.exp(-decay_rate * age_days)

        # Combine base score with recency boost
        # Formula: final_score = base_score * (1 + weight * recency_factor)
        boosted_score = base_score * (1 + boost_weight * recency_factor)

        boosted_nodes.append(NodeWithScore(node=node.node, score=boosted_score))

    # Sort by boosted score (descending)
    boosted_nodes.sort(key=lambda x: x.score or 0, reverse=True)

    return boosted_nodes


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
        # Never log user queries (may contain secrets).
        
        # Preprocess query to improve retrieval quality
        enhanced_query = _preprocess_query(query)
        
        # Load index
        index = load_llamaindex_index()

        # Use retriever (no LLM needed - just retrieval)
        retriever = index.as_retriever(similarity_top_k=RETRIEVAL_EMBED_TOP_K)

        # Retrieve relevant chunks (embedding stage) using enhanced query
        nodes = retriever.retrieve(enhanced_query)

        # Search Confluence in parallel (with timeout)
        # Optimization: Check if CONFLUENCE_URL is set before even attempting it
        confluence_nodes = []
        if os.getenv("CONFLUENCE_URL"):
            try:
                 # Run blocking generic search in executor
                 # Use enhanced_query (preprocessed) instead of original query for consistency
                 loop = asyncio.get_running_loop()
                 result = await asyncio.wait_for(
                     loop.run_in_executor(None, _search_confluence, enhanced_query),
                     timeout=CONFLUENCE_TIMEOUT
                 )
                 # Always assign result, even if empty list
                 # (Previously: "if result:" would skip assignment for empty lists, which was redundant
                 #  since confluence_nodes is already initialized as []. This ensures Confluence
                 #  is always called when CONFLUENCE_URL is set, regardless of result.)
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
            nodes = _filter_nodes_by_date(nodes, date_from, date_to)
            logger.info(f"Date filtering: {original_count} -> {len(nodes)} nodes")

        # Apply recency boost if configured
        if RETRIEVAL_RECENCY_BOOST > 0:
            nodes = _apply_recency_boost(nodes, RETRIEVAL_RECENCY_BOOST, RETRIEVAL_RECENCY_HALF_LIFE_DAYS)

        # Rerank the retrieved nodes with FlashRank (CPU-only, ONNX-based) and trim to the
        # configured final Top K for the tool response.
        rerank_scores: dict[int, float] = {}
        if nodes:
            rerank_limit = max(1, min(RETRIEVAL_RERANK_TOP_K, len(nodes)))
            try:
                reranker = _ensure_reranker()
                # Prepare documents for reranking - FlashRank expects list of dicts with 'text' key
                passages = [{"text": node.node.get_content() or ""} for node in nodes]
                
                # FlashRank requires a RerankRequest object
                from flashrank import RerankRequest
                rerank_request = RerankRequest(query=enhanced_query, passages=passages)
                
                # FlashRank returns reranked documents (list of dicts with 'text' and 'score')
                reranked_results = reranker.rerank(rerank_request)
                
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
                    # FlashRank returns dicts with 'text' and 'score' keys
                    doc_text = result.get("text", "")
                    score = result.get("score", 0.0)
                    
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
            except Exception as e:
                # Fall back to vector search ordering if reranking fails
                logger.error(f"Reranking failed, falling back to vector search order: {e}")
                nodes = nodes[:rerank_limit]

        # Filter nodes by score threshold
        if RETRIEVAL_SCORE_THRESHOLD > 0:
            nodes = [
                node for node in nodes
                if rerank_scores.get(id(node), 0.0) >= RETRIEVAL_SCORE_THRESHOLD
            ]

        # Format chunks with simplified structure
        chunks = []
        for node in nodes:
            metadata = dict(node.node.metadata or {})
            chunk_text = node.node.get_content()

            # Get file path for URI building
            file_path = (
                metadata.get("file_path")
                or metadata.get("file_name")
                or metadata.get("source")
            )
            original_source = metadata.get("source")

            # Build heading path from document structure
            headings = metadata.get("document_headings") or metadata.get("headings") or []
            char_start = getattr(node.node, "start_char_idx", None)
            heading_text = metadata.get("heading")
            heading_path: list[str] = []
            if isinstance(headings, list) and headings:
                if heading_text is None and char_start is not None:
                    heading_text, heading_path = _build_heading_path(headings, char_start)
            # Use explicit heading_path from metadata if available
            meta_heading_path = metadata.get("heading_path")
            if not heading_path and meta_heading_path:
                heading_path = list(meta_heading_path)
            # Add current heading to path if not already included
            if heading_text and (not heading_path or heading_path[-1] != heading_text):
                heading_path = heading_path + [heading_text] if heading_path else [heading_text]

            # Build URI for the source document
            source_uri = None
            if original_source == "Confluence":
                # Handle Confluence sources
                confluence_url = os.getenv("CONFLUENCE_URL", "")
                page_id = metadata.get("page_id")
                if confluence_url and page_id:
                    source_uri = f"{confluence_url.rstrip('/')}/pages/viewpage.action?pageId={page_id}"
                elif confluence_url:
                    title = metadata.get("title", metadata.get("file_name", ""))
                    if title:
                        from urllib.parse import quote
                        encoded_title = quote(title.replace(" ", "+"))
                        source_uri = f"{confluence_url.rstrip('/')}/spaces/~{encoded_title}"
            elif file_path:
                # Create file:// URI for local files
                try:
                    file_path_obj = Path(str(file_path))
                    if file_path_obj.is_absolute():
                        source_uri = f"file://{file_path_obj.resolve()}"
                    else:
                        data_dir = Path(os.getenv("DATA_DIR", "./data"))
                        resolved_path = (data_dir / file_path_obj).resolve()
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
                line_number = _char_offset_to_line(char_start, line_offsets)

            # Build simplified location object
            location = {
                "uri": source_uri,
                "page": page_number,
                "line": line_number,
                "heading_path": heading_path if heading_path else None,
            }

            # Build chunk with simplified structure (no metadata)
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

