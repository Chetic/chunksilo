"""Reranking model management for FlashRank."""

import logging
from pathlib import Path
from typing import Optional

import opd_mcp.config as config

logger = logging.getLogger(__name__)

# Global cache for reranker model instance
_reranker_model = None


def _map_model_name(model_name: str) -> str:
    """Map cross-encoder model names to FlashRank equivalents.

    FlashRank supports models like 'ms-marco-TinyBERT-L-2-v2', 'ms-marco-MiniLM-L-12-v2'.
    Note: FlashRank doesn't have L-6 models, so we map to L-12 equivalents.

    Args:
        model_name: Original model name

    Returns:
        FlashRank-compatible model name
    """
    model_mapping = {
        "cross-encoder/ms-marco-MiniLM-L-6-v2": "ms-marco-MiniLM-L-12-v2",
        "ms-marco-MiniLM-L-6-v2": "ms-marco-MiniLM-L-12-v2",
    }

    if model_name in model_mapping:
        return model_mapping[model_name]
    elif model_name.startswith("cross-encoder/"):
        base_name = model_name.replace("cross-encoder/", "")
        if "L-6" in base_name:
            return base_name.replace("L-6", "L-12")
        return base_name

    return model_name


def ensure_reranker(
    cache_dir: Path = None, offline: bool = None, model_name: str = None
):
    """Load and return the FlashRank reranking model.

    Args:
        cache_dir: Directory for model cache (defaults to RETRIEVAL_MODEL_CACHE_DIR)
        offline: Whether to use offline mode (defaults to is_offline_mode())
        model_name: Model name (defaults to RETRIEVAL_RERANK_MODEL_NAME)

    Returns:
        FlashRank Ranker instance

    Raises:
        ImportError: If flashrank is not installed
        FileNotFoundError: If offline mode is enabled but model is not cached
    """
    global _reranker_model

    if _reranker_model is not None:
        return _reranker_model

    try:
        from flashrank import Ranker
    except ImportError as exc:
        raise ImportError(
            "flashrank is required for reranking. Please install dependencies from requirements.txt."
        ) from exc

    cache_dir = cache_dir or config.RETRIEVAL_MODEL_CACHE_DIR
    offline = offline if offline is not None else config.is_offline_mode()
    model_name = _map_model_name(model_name or config.RETRIEVAL_RERANK_MODEL_NAME)

    try:
        _reranker_model = Ranker(model_name=model_name, cache_dir=str(cache_dir))
    except Exception as exc:
        if offline:
            raise FileNotFoundError(
                f"Rerank model '{model_name}' not available in cache directory {cache_dir}. "
                "Download it before running in offline mode."
            ) from exc
        raise

    logger.info(f"Rerank model '{model_name}' loaded successfully")
    return _reranker_model


def ensure_rerank_model_cached(cache_dir: Path, offline: bool = False) -> Path:
    """Ensure the reranking model is cached locally.

    Args:
        cache_dir: Directory for model cache
        offline: Whether to use offline mode

    Returns:
        The cache directory path

    Raises:
        ImportError: If flashrank is not installed
        FileNotFoundError: If offline mode is enabled but model is not cached
    """
    try:
        from flashrank import Ranker
    except ImportError as exc:
        raise ImportError("flashrank is required for reranking.") from exc

    cache_dir_abs = cache_dir.resolve()
    logger.info("Ensuring rerank model is available in cache...")

    model_name = _map_model_name(config.RETRIEVAL_RERANK_MODEL_NAME)

    try:
        Ranker(model_name=model_name, cache_dir=str(cache_dir_abs))
        logger.info(f"FlashRank model '{model_name}' initialized successfully")
        return cache_dir_abs
    except Exception as exc:
        if offline:
            raise FileNotFoundError(
                f"Rerank model '{model_name}' not found in cache."
            ) from exc
        raise


def reset_reranker():
    """Reset the cached reranker model (for testing)."""
    global _reranker_model
    _reranker_model = None


def get_reranker():
    """Get the current reranker model (may be None if not initialized)."""
    return _reranker_model
