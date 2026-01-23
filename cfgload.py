#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Shared configuration loading for ChunkSilo.

Loads configuration from config.yaml.
"""
import yaml
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(__file__).parent / "config.yaml"

_DEFAULTS: dict[str, Any] = {
    "indexing": {
        "directories": ["./data"],
        "defaults": {
            "include": ["**/*.pdf", "**/*.md", "**/*.txt", "**/*.docx", "**/*.doc"],
            "exclude": [],
            "recursive": True,
        },
        "chunk_size": 1600,
        "chunk_overlap": 200,
    },
    "retrieval": {
        "embed_model_name": "BAAI/bge-small-en-v1.5",
        "embed_top_k": 20,
        "rerank_model_name": "ms-marco-MiniLM-L-12-v2",
        "rerank_top_k": 5,
        "rerank_candidates": 100,
        "score_threshold": 0.1,
        "recency_boost": 0.3,
        "recency_half_life_days": 365,
        "bm25_similarity_top_k": 10,
        "offline": True,
    },
    "confluence": {
        "url": "",
        "username": "",
        "api_token": "",
        "timeout": 10.0,
        "max_results": 30,
    },
    "ssl": {
        "ca_bundle_path": "",
    },
    "storage": {
        "storage_dir": "./storage",
        "model_cache_dir": "./models",
    },
}

_config_cache: dict[str, Any] | None = None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override into base, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file with defaults.

    Args:
        config_path: Optional path to config file. If None, uses default CONFIG_PATH.
                    Results are cached only when using the default path.

    Returns:
        Configuration dictionary with defaults merged in.
    """
    global _config_cache

    # Return cached config if available (only for default path)
    if _config_cache is not None and config_path is None:
        return _config_cache

    path = config_path or CONFIG_PATH

    if not path.exists():
        # Return defaults if config file doesn't exist
        return _DEFAULTS.copy()

    with open(path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    result = _deep_merge(_DEFAULTS, user_config)

    # Cache result only for default path
    if config_path is None:
        _config_cache = result

    return result


def get(key: str, default: Any = None) -> Any:
    """Get a config value by dot-notation key.

    Args:
        key: Dot-separated key path (e.g., 'retrieval.embed_top_k')
        default: Value to return if key not found

    Returns:
        Configuration value or default.

    Example:
        >>> get('retrieval.embed_top_k')
        20
        >>> get('storage.storage_dir')
        './storage'
    """
    config = load_config()
    keys = key.split(".")
    value: Any = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    return value


def reload_config() -> dict[str, Any]:
    """Force reload configuration from disk, clearing the cache."""
    global _config_cache
    _config_cache = None
    return load_config()
