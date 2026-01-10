"""Centralized configuration for OPD-MCP.

All environment variables and configuration constants are defined here
to ensure consistency between ingestion and retrieval components.
"""

import os
from pathlib import Path

# ==============================================================================
# Directory Configuration
# ==============================================================================

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "./storage"))
STATE_DB_PATH = STORAGE_DIR / "ingestion_state.db"
HEADING_STORE_PATH = STORAGE_DIR / "heading_store.json"
BM25_INDEX_DIR = STORAGE_DIR / "bm25_index"

# ==============================================================================
# Model Configuration
# ==============================================================================

# Shared cache directory for embedding and reranking models
RETRIEVAL_MODEL_CACHE_DIR = Path(os.getenv("RETRIEVAL_MODEL_CACHE_DIR", "./models"))

# Stage 1: Vector search over embeddings
RETRIEVAL_EMBED_MODEL_NAME = os.getenv(
    "RETRIEVAL_EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5"
)
RETRIEVAL_EMBED_TOP_K = int(os.getenv("RETRIEVAL_EMBED_TOP_K", "20"))

# Stage 2: FlashRank reranking (CPU-only, ONNX-based)
RETRIEVAL_RERANK_MODEL_NAME = os.getenv(
    "RETRIEVAL_RERANK_MODEL_NAME", "ms-marco-MiniLM-L-12-v2"
)
RETRIEVAL_RERANK_TOP_K = int(os.getenv("RETRIEVAL_RERANK_TOP_K", "5"))

# ==============================================================================
# Chunking Configuration
# ==============================================================================

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# ==============================================================================
# Retrieval Configuration
# ==============================================================================

# BM25 file name search
BM25_SIMILARITY_TOP_K = int(os.getenv("BM25_SIMILARITY_TOP_K", "10"))

# Recency boost
RETRIEVAL_RECENCY_BOOST = float(os.getenv("RETRIEVAL_RECENCY_BOOST", "0.3"))
RETRIEVAL_RECENCY_HALF_LIFE_DAYS = int(os.getenv("RETRIEVAL_RECENCY_HALF_LIFE_DAYS", "365"))

# Score threshold for filtering low-relevance results
RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.1"))

# ==============================================================================
# Confluence Configuration
# ==============================================================================

CONFLUENCE_TIMEOUT = float(os.getenv("CONFLUENCE_TIMEOUT", "10.0"))
CONFLUENCE_MAX_RESULTS = int(os.getenv("CONFLUENCE_MAX_RESULTS", "30"))

# Common English stopwords to filter from Confluence CQL queries
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

# ==============================================================================
# SSL/TLS Configuration
# ==============================================================================

CA_BUNDLE_PATH = os.getenv("CA_BUNDLE_PATH")

# ==============================================================================
# Metadata Exclusion Configuration
# ==============================================================================

# These keys are excluded from the embedding text to save tokens
EXCLUDED_EMBED_METADATA_KEYS = [
    "line_offsets",       # Large integer array, primary cause of length errors
    "document_headings",  # Heading hierarchy array with positions
    "heading_path",       # Pre-computed heading hierarchy, stored separately
    "file_path",          # Redundant with file_name/source
    "source",             # Often same as file_path
    "creation_date",      # Temporal, not semantic
    "last_modified_date", # Temporal, not semantic
    "doc_ids",            # Internal tracking
    "hash",               # Internal tracking
]

# These keys are excluded from the LLM context to save context window
EXCLUDED_LLM_METADATA_KEYS = [
    "line_offsets",  # LLM needs text content, not integer map
    "hash",          # Internal tracking
    "doc_ids",       # Internal tracking
    "file_path",     # Usually redundant if file_name is present
    "source",        # Usually redundant
]

# ==============================================================================
# Offline Mode Configuration
# ==============================================================================


def configure_offline_mode(offline: bool, cache_dir: Path | None = None) -> bool:
    """Configure environment variables for offline mode.

    Args:
        offline: Whether to enable offline mode
        cache_dir: Optional cache directory (defaults to RETRIEVAL_MODEL_CACHE_DIR)

    Returns:
        Whether offline mode is enabled
    """
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

        cache_dir = cache_dir or RETRIEVAL_MODEL_CACHE_DIR
        cache_dir_abs = cache_dir.resolve()
        os.environ["HF_HOME"] = str(cache_dir_abs)
        os.environ["HF_HUB_CACHE"] = str(cache_dir_abs)
        os.environ["HF_DATASETS_CACHE"] = str(cache_dir_abs)
    return offline


def configure_ssl(logger=None):
    """Configure SSL/TLS settings from CA_BUNDLE_PATH.

    Args:
        logger: Optional logger for info/warning messages
    """
    if CA_BUNDLE_PATH:
        ca_bundle_path = Path(CA_BUNDLE_PATH)
        if ca_bundle_path.exists():
            os.environ["REQUESTS_CA_BUNDLE"] = str(ca_bundle_path.resolve())
            os.environ["SSL_CERT_FILE"] = str(ca_bundle_path.resolve())
            if logger:
                logger.info(f"CA bundle configured: {ca_bundle_path.resolve()}")
        else:
            if logger:
                logger.warning(f"CA bundle path does not exist: {ca_bundle_path.resolve()}")


def is_offline_mode() -> bool:
    """Check if offline mode is enabled from environment."""
    return os.getenv("OFFLINE", "1").lower() not in ("0", "false", "no")
