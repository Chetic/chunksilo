#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared model utilities used by both the indexing and search pipelines."""
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_cached_model_path(cache_dir: Path, model_name: str) -> Path | None:
    """Get the cached model directory path using huggingface_hub's snapshot_download."""
    try:
        from fastembed import TextEmbedding
        from huggingface_hub import snapshot_download
        models = TextEmbedding.list_supported_models()
        model_info = [m for m in models if m.get("model") == model_name]
        if model_info:
            hf_source = model_info[0].get("sources", {}).get("hf")
            if hf_source:
                cache_dir_abs = cache_dir.resolve()
                model_dir = snapshot_download(
                    repo_id=hf_source,
                    local_files_only=True,
                    cache_dir=str(cache_dir_abs)
                )
                return Path(model_dir).resolve()
    except (ImportError, Exception):
        logger.debug("Could not resolve cached model path for %s", model_name, exc_info=True)
    return None


def resolve_flashrank_model_name(model_name: str) -> str:
    """Map cross-encoder model names to FlashRank equivalents.

    FlashRank doesn't have L-6 models, so we map to L-12 equivalents.
    The cross-encoder/ prefix is also stripped since FlashRank uses bare names.
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
        else:
            return base_name
    return model_name


def configure_offline_mode(offline: bool, cache_dir: Path) -> None:
    """Configure environment variables for offline mode."""
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        cache_dir_abs = cache_dir.resolve()
        os.environ["HF_HOME"] = str(cache_dir_abs)
        os.environ["HF_HUB_CACHE"] = str(cache_dir_abs)
        os.environ["HF_DATASETS_CACHE"] = str(cache_dir_abs)
        logger.info("Offline mode enabled.")
    else:
        # Clear offline mode environment variables to allow downloads
        for var in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]:
            os.environ.pop(var, None)

    # Update huggingface_hub's cached constant (it caches at import time)
    try:
        from huggingface_hub import constants
        constants.HF_HUB_OFFLINE = offline
    except ImportError:
        logger.debug("huggingface_hub.constants not available for offline patching")
