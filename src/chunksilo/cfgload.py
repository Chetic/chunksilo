#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Shared configuration loading for ChunkSilo.

Loads configuration from config.yaml, searching in standard locations. The
first load with an explicit path (``--config``) becomes the process-wide
active configuration: every later no-argument ``load_config()`` or ``get()``
call answers from it, so a path given on the command line governs the whole
process rather than only the call sites it was threaded through.
"""
import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Retrieval models bundled inside the installed package (populated at wheel-build
# time). Present in air-gapped wheel installs; absent in plain source checkouts.
_BUNDLED_MODELS_DIR = Path(__file__).resolve().parent / "_bundled_models"


def _find_config() -> Path:
    """Find config.yaml using a priority-based search.

    Search order:
    1. CHUNKSILO_CONFIG environment variable
    2. ./config.yaml (current working directory)
    3. ~/.config/chunksilo/config.yaml (XDG standard)
    """
    env_path = os.environ.get("CHUNKSILO_CONFIG")
    if env_path:
        return Path(env_path)

    cwd_path = Path.cwd() / "config.yaml"
    if cwd_path.exists():
        return cwd_path

    xdg_path = Path.home() / ".config" / "chunksilo" / "config.yaml"
    if xdg_path.exists():
        return xdg_path

    # Return cwd path as default (will fall through to defaults if not found)
    return cwd_path


CONFIG_PATH = _find_config()

# Default file type patterns - the single definition; index.py imports these.
DEFAULT_INCLUDE_PATTERNS = ["**/*.pdf", "**/*.md", "**/*.txt", "**/*.docx", "**/*.doc"]

DEFAULT_EXCLUDE_PATTERNS = [
    "**/.git/**",
    "**/node_modules/**",
    "**/__pycache__/**",
    "**/.venv/**",
    "**/venv/**",
    "**/.tox/**",
    "**/.mypy_cache/**",
    "**/.pytest_cache/**",
    "**/.eggs/**",
    "**/*.egg-info/**",
    "**/.DS_Store",
]

# Filename markers of copies made to collect review comments, matched
# case-insensitively against the filename without its extension. Setting
# indexing.versioning.review_copy_patterns replaces this list wholesale;
# [] turns detection off.
DEFAULT_REVIEW_COPY_PATTERNS = [
    r"\breview\b",
    r"\bcomments?\b",
    r"\bkommentar(?:er)?\b",
    r"\bgranskning\b",
    r"\bcopy\b",  # "Copy of Spec", "Spec - Copy (2)"
    r"\bkopia\b",  # "Kopia av Spec", "Spec - kopia"
]

_DEFAULTS: dict[str, Any] = {
    "indexing": {
        "directories": ["./data"],
        "defaults": {
            "include": DEFAULT_INCLUDE_PATTERNS,
            "exclude": DEFAULT_EXCLUDE_PATTERNS,
            "recursive": True,
            "case_sensitive": False,
        },
        "chunk_size": 512,
        "chunk_overlap": 50,
        "batch_size": 200,  # upper bound; adaptive sizing may shrink a batch
        "parallel_workers": 8,  # 1 = load files serially
        "checkpoint_interval_files": 500,
        "checkpoint_interval_seconds": 300,
        "per_file_seconds": 300,  # give up on a single file; 0 disables
        "scan_item_seconds": 30,  # timeout for stat/hash/walk during scanning
        "slow_file_threshold_seconds": 30,  # warn when one file takes longer
        "versioning": {
            # Group files that are revisions of one document (revision-named
            # directories or filename tokens like "Spec PB3.docx") and index
            # only the newest member of each group, by modification time.
            "enabled": True,
            # Regexes tried in order against the full path; files sharing an
            # extracted document ID form one group. The first capture group
            # (or the whole match) is the ID. Empty = group by normalized
            # name/path only.
            "doc_id_patterns": [],
            # Extra revision-token regex fragments (case-insensitive), added
            # to the built-in grammar for directory names and filename
            # suffixes alike.
            "extra_revision_tokens": [],
            "review_copy_patterns": DEFAULT_REVIEW_COPY_PATTERNS,
            # Index older revisions too, stamped superseded and ranked below
            # current documents in results.
            "index_superseded": False,
        },
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
        # At most this many chunks of one document in the final results;
        # freed slots go to the next-best other documents. 0 disables.
        "max_chunks_per_doc": 2,
        "offline": False,
    },
    "confluence": {
        "url": "",
        "username": "",
        "api_token": "",
        "timeout": 10.0,
        "max_results": 30,
    },
    "jira": {
        "url": "",
        "username": "",
        "api_token": "",
        "timeout": 10.0,
        "max_results": 30,
        "projects": [],  # Empty list = all accessible projects
        "include_comments": False,
        "include_custom_fields": False,
    },
    "ssl": {
        # CA bundle for environments with TLS interception or a private CA:
        # used for model downloads and for the Confluence/Jira APIs.
        "ca_bundle_path": "",
    },
    "storage": {
        "storage_dir": "./storage",
        "model_cache_dir": "./models",
    },
    "server": {
        "transport": "stdio",  # stdio | streamable-http
        # Loopback by default. The HTTP transport performs no authentication
        # of its own: exposing the port is a deliberate act, taken together
        # with TLS termination and a login layer in front of this server.
        "host": "127.0.0.1",
        "port": 8400,
    },
    # Present files indexed from a mounted network share at the share's own
    # location (an smb:// URI plus the Windows UNC path) instead of the
    # server-local mount path, so a client on another machine gets a location
    # it can open. Uncovered files keep their file:// URIs.
    # [{prefix: "/mnt/docs", unc: "//nas/docs"}]
    "shares": [],
}

# Options that no longer exist, mapped to what replaced them. A config file
# that still sets one gets a warning instead of silently falling back to the
# built-in default.
_REMOVED_KEYS = {
    "indexing.timeout": "indexing.per_file_seconds (0 disables) and indexing.scan_item_seconds",
    "indexing.logging": "indexing.slow_file_threshold_seconds",
    "indexing.enable_parallel_loading": "indexing.parallel_workers (1 = serial)",
    "indexing.enable_adaptive_batching": "indexing.batch_size (an upper bound; sizing adapts automatically)",
    "indexing.max_memory_mb": "indexing.batch_size (an upper bound; sizing adapts automatically)",
    "retrieval.bm25_similarity_top_k": "nothing; the number of file-name matches is fixed",
}

# The active configuration: the merged result of the last load, plus the path
# it came from. An explicit-path load replaces it; no-argument loads reuse it.
_config_cache: dict[str, Any] | None = None
_active_path: Path | None = None


def _warn_removed_keys(user_config: dict[str, Any], path: Path) -> None:
    """Warn about options the file sets that the schema no longer has."""
    for dotted, replacement in _REMOVED_KEYS.items():
        section, _, key = dotted.partition(".")
        block = user_config.get(section)
        if isinstance(block, dict) and key in block:
            logger.warning(
                "%s: '%s' is no longer used and is ignored; use %s instead",
                path, dotted, replacement,
            )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override into base, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if value is None and key in result:
            # An empty YAML value ("chunk_size:" or a commented-out section)
            # parses as None; keep the default instead of poisoning it.
            continue
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _has_embedding_cache(cache_dir: Path) -> bool:
    """True if ``cache_dir`` holds a HuggingFace-style embedding model cache."""
    try:
        return any(cache_dir.glob("models--*"))
    except OSError:
        return False


def _resolve_bundled_models(config: dict[str, Any]) -> dict[str, Any]:
    """Transparently fall back to models bundled in the package.

    If the wheel ships the retrieval models and the configured cache directory
    does not already contain them, point the cache at the bundled copy and enable
    offline mode. This lets a pip-installed wheel run in an air-gapped environment
    with zero extra configuration. A user-configured cache that already holds the
    models always wins, so explicit setups are never overridden.

    Detection looks for any HuggingFace-style ``models--*`` cache directory rather
    than matching the configured embedding name: fastembed stores the model under
    its upstream source repo (e.g. ``models--qdrant--bge-small-en-v1.5-onnx-q``),
    not under the configured alias ``BAAI/bge-small-en-v1.5``.
    """
    if not _has_embedding_cache(_BUNDLED_MODELS_DIR):
        return config  # No models bundled in this install.

    configured = Path(config["storage"]["model_cache_dir"]).expanduser()
    if configured.resolve() != _BUNDLED_MODELS_DIR.resolve() and _has_embedding_cache(
        configured
    ):
        return config  # Configured cache already has models; respect it.

    result = copy.deepcopy(config)
    result["storage"]["model_cache_dir"] = str(_BUNDLED_MODELS_DIR)
    result["retrieval"]["offline"] = True
    logger.info(
        "Using bundled retrieval models at %s (offline mode)", _BUNDLED_MODELS_DIR
    )
    return result


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load configuration from YAML file with defaults.

    Args:
        config_path: Optional path to config file. When given, the result
                    becomes the process-wide active configuration that
                    no-argument calls (and :func:`get`) answer from. When
                    None, returns the active configuration, loading from the
                    auto-discovered CONFIG_PATH on first use.

    Returns:
        Configuration dictionary with defaults merged in.
    """
    global _config_cache, _active_path

    if _config_cache is not None and (
        config_path is None or Path(config_path) == _active_path
    ):
        return _config_cache

    path = Path(config_path) if config_path else CONFIG_PATH

    # Callers may mutate the returned dict; never hand out the defaults themselves.
    defaults = copy.deepcopy(_DEFAULTS)

    if not path.exists():
        logger.info("Config file not found at %s; using built-in defaults", path)
        result = _resolve_bundled_models(defaults)
    else:
        logger.info("Using config: %s", path)
        with open(path, encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        _warn_removed_keys(user_config, path)
        result = _deep_merge(defaults, user_config)
        result = _resolve_bundled_models(result)

    _config_cache = result
    _active_path = path
    return result


def get(key: str, default: Any = None) -> Any:
    """Get a value from the active configuration by dot-notation key.

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
    """Force reload configuration from disk, clearing the active config."""
    global _config_cache, _active_path
    _config_cache = None
    _active_path = None
    return load_config()
