"""Embedding model management for FastEmbed."""

import logging
from pathlib import Path
from typing import Optional

from llama_index.core import Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding

from opd_mcp.config import (
    RETRIEVAL_EMBED_MODEL_NAME,
    RETRIEVAL_MODEL_CACHE_DIR,
    is_offline_mode,
)

logger = logging.getLogger(__name__)

# Global cache for embedding model initialization state
_embed_model_initialized = False


def get_cached_model_path(cache_dir: Path, model_name: str) -> Optional[Path]:
    """Get the cached model directory path using huggingface_hub's snapshot_download.

    This works completely offline and bypasses fastembed's download_model API calls.

    Args:
        cache_dir: Directory where models are cached
        model_name: FastEmbed model name (e.g., 'BAAI/bge-small-en-v1.5')

    Returns:
        Path to the cached model directory, or None if not found
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
                model_dir = snapshot_download(
                    repo_id=hf_source,
                    local_files_only=True,
                    cache_dir=str(cache_dir_abs),
                )
                return Path(model_dir).resolve()
    except (ImportError, Exception):
        pass
    return None


def verify_model_cache_exists(cache_dir: Path, model_name: str = None) -> bool:
    """Verify that the cached model directory exists and contains expected files.

    Args:
        cache_dir: Directory where models are cached
        model_name: Model name to check (defaults to RETRIEVAL_EMBED_MODEL_NAME)

    Returns:
        True if model cache exists and is valid
    """
    from fastembed import TextEmbedding

    model_name = model_name or RETRIEVAL_EMBED_MODEL_NAME

    try:
        models = TextEmbedding.list_supported_models()
        model_info = [m for m in models if m.get("model") == model_name]
        if not model_info:
            return False

        model_info = model_info[0]
        hf_source = model_info.get("sources", {}).get("hf")
        if not hf_source:
            return False

        expected_dir = cache_dir / f"models--{hf_source.replace('/', '--')}"
        if not expected_dir.exists():
            return False

        snapshots_dir = expected_dir / "snapshots"
        if not snapshots_dir.exists():
            return False

        model_file = model_info.get("model_file", "model_optimized.onnx")
        for snapshot in snapshots_dir.iterdir():
            if snapshot.is_dir():
                model_path = snapshot / model_file
                if model_path.exists() or model_path.is_symlink():
                    return True

        return False
    except Exception:
        return False


def create_fastembed_embedding(
    cache_dir: Path, offline: bool = False, model_name: str = None
) -> FastEmbedEmbedding:
    """Create a FastEmbedEmbedding instance.

    Args:
        cache_dir: Directory for model cache
        offline: Whether to use offline mode
        model_name: Model name (defaults to RETRIEVAL_EMBED_MODEL_NAME)

    Returns:
        Configured FastEmbedEmbedding instance
    """
    model_name = model_name or RETRIEVAL_EMBED_MODEL_NAME

    if offline:
        cached_model_path = get_cached_model_path(cache_dir, model_name)
        if cached_model_path:
            logger.info(f"Using cached model path to bypass download: {cached_model_path}")
            return FastEmbedEmbedding(
                model_name=model_name,
                cache_dir=str(cache_dir),
                specific_model_path=str(cached_model_path),
            )
        else:
            logger.warning(
                "Could not find cached model path, falling back to normal initialization"
            )

    return FastEmbedEmbedding(model_name=model_name, cache_dir=str(cache_dir))


def ensure_embed_model(
    cache_dir: Path = None, offline: bool = None, model_name: str = None
) -> None:
    """Ensure the embedding model is configured in LlamaIndex Settings.

    This must be called before using the index to ensure query embeddings
    use the same model as ingestion (FastEmbed, not OpenAI).

    Args:
        cache_dir: Directory for model cache (defaults to RETRIEVAL_MODEL_CACHE_DIR)
        offline: Whether to use offline mode (defaults to is_offline_mode())
        model_name: Model name (defaults to RETRIEVAL_EMBED_MODEL_NAME)
    """
    global _embed_model_initialized

    if _embed_model_initialized:
        return

    cache_dir = cache_dir or RETRIEVAL_MODEL_CACHE_DIR
    offline = offline if offline is not None else is_offline_mode()
    model_name = model_name or RETRIEVAL_EMBED_MODEL_NAME

    cached_model_path = get_cached_model_path(cache_dir, model_name)
    if cached_model_path and offline:
        logger.info(f"Loading embedding model from cache: {cached_model_path}")
        embed_model = FastEmbedEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir),
            specific_model_path=str(cached_model_path),
        )
    else:
        embed_model = FastEmbedEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir),
        )

    logger.info("Embedding model initialized successfully")
    Settings.embed_model = embed_model
    _embed_model_initialized = True


def ensure_embedding_model_cached(cache_dir: Path, offline: bool = False) -> None:
    """Ensure the embedding model is available in the local cache.

    Args:
        cache_dir: Directory for model cache
        offline: Whether to use offline mode

    Raises:
        FileNotFoundError: If offline mode is enabled but model is not cached
        RuntimeError: If model download/initialization fails
    """
    import os

    if offline:
        logger.info("Verifying embedding model cache...")
        if verify_model_cache_exists(cache_dir):
            logger.info("Embedding model found in cache")
        else:
            logger.error(
                "Offline mode enabled, but embedding model cache not found in %s",
                cache_dir,
            )
            raise FileNotFoundError(
                f"Embedding model '{RETRIEVAL_EMBED_MODEL_NAME}' not found in cache directory '{cache_dir}'."
            )

    try:
        logger.info("Initializing embedding model from cache...")
        cache_dir_abs = cache_dir.resolve()
        if offline:
            os.environ["HF_HUB_CACHE"] = str(cache_dir_abs)

        create_fastembed_embedding(cache_dir, offline=offline)
        logger.info("Embedding model initialized successfully")
        return
    except (ValueError, Exception) as e:
        if offline:
            raise FileNotFoundError(f"Failed to load model offline: {e}") from e
        else:
            raise RuntimeError(f"Failed to download/initialize model: {e}") from e


def reset_embed_model_initialized():
    """Reset the embedding model initialization state (for testing)."""
    global _embed_model_initialized
    _embed_model_initialized = False


def is_embed_model_initialized() -> bool:
    """Check if the embedding model has been initialized."""
    return _embed_model_initialized
