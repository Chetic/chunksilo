#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Core search pipeline for ChunkSilo.

Contains all retrieval logic independent of the MCP server.
Used by both the MCP server (server.py) and the CLI (cli.py).
"""
import os
import re
import time
import math
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path
from typing import Any

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from .index import get_heading_store
try:
    from llama_index.readers.confluence import ConfluenceReader
    import requests  # Available when llama-index-readers-confluence is installed
except ImportError:
    ConfluenceReader = None
    requests = None

# Optional Jira integration
try:
    from jira import JIRA
except ImportError:
    JIRA = None

# TEMPORARY FIX: Patch Confluence HTML parser to handle syntax highlighting spans
# Remove when upstream issue is fixed (see confluence_html_formatter.py)
if ConfluenceReader is not None:
    try:
        from .confluence_html_formatter import patch_confluence_reader
        patch_confluence_reader()
    except ImportError:
        pass

from .cfgload import load_config

logger = logging.getLogger(__name__)


def _init_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load and return configuration."""
    return load_config(config_path)


# Module-level config (loaded on first use)
_config: dict[str, Any] | None = None


def _get_config() -> dict[str, Any]:
    """Get the module config, loading defaults if not yet initialized."""
    global _config
    if _config is None:
        _config = _init_config()
    return _config


# Global caches
_index_cache = None
_embed_model_initialized = False
_reranker_model = None
_bm25_retriever_cache = None
_configured_directories_cache: list[Path] | None = None

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


def _setup_offline_mode(config: dict[str, Any]) -> None:
    """Configure offline mode for HuggingFace libraries if enabled."""
    offline_mode = config["retrieval"]["offline"]
    if offline_mode:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        cache_dir_abs = Path(config["storage"]["model_cache_dir"]).resolve()
        os.environ["HF_HOME"] = str(cache_dir_abs)
        os.environ["HF_HUB_CACHE"] = str(cache_dir_abs)
        os.environ["HF_DATASETS_CACHE"] = str(cache_dir_abs)


def _setup_ssl(config: dict[str, Any]) -> str | None:
    """Configure SSL/TLS CA bundle if specified. Returns the CA bundle path or None."""
    ca_bundle_path = config["ssl"]["ca_bundle_path"] or None
    if ca_bundle_path:
        ca_path = Path(ca_bundle_path)
        if ca_path.exists():
            os.environ["REQUESTS_CA_BUNDLE"] = str(ca_path.resolve())
            os.environ["SSL_CERT_FILE"] = str(ca_path.resolve())
            logger.info(f"CA bundle configured: {ca_path.resolve()}")
        else:
            logger.warning(f"CA bundle path does not exist: {ca_path.resolve()}")
    return ca_bundle_path


def _get_configured_directories(config: dict[str, Any]) -> list[Path]:
    """Get list of configured data directories for path resolution."""
    global _configured_directories_cache

    if _configured_directories_cache is not None:
        return _configured_directories_cache

    dirs: list[Path] = []
    for entry in config.get("indexing", {}).get("directories", []):
        if isinstance(entry, str):
            dirs.append(Path(entry))
        elif isinstance(entry, dict) and entry.get("enabled", True):
            path_str = entry.get("path")
            if path_str:
                dirs.append(Path(path_str))

    _configured_directories_cache = dirs if dirs else []
    return _configured_directories_cache


def _resolve_file_uri(file_path: str, config: dict[str, Any]) -> str | None:
    """Resolve a file path to a file:// URI."""
    try:
        file_path_obj = Path(str(file_path))

        if file_path_obj.is_absolute():
            if file_path_obj.exists():
                return f"file://{file_path_obj.resolve()}"
            return f"file://{file_path_obj}"

        for data_dir in _get_configured_directories(config):
            candidate = data_dir / file_path_obj
            if candidate.exists():
                return f"file://{candidate.resolve()}"

        return f"file://{file_path_obj.resolve()}"
    except Exception:
        return None


def _build_heading_path(headings: list[dict], char_start: int | None) -> tuple[str | None, list[str]]:
    """Build a heading path for the given character position within a document."""
    if not headings or char_start is None:
        return None, []

    current_idx = None
    for idx, heading in enumerate(headings):
        heading_pos = heading.get("position", 0)
        if heading_pos <= char_start:
            current_idx = idx
        else:
            break

    if current_idx is None:
        return None, []

    path = [h.get("text", "") for h in headings[: current_idx + 1] if h.get("text")]
    current_heading_text = path[-1] if path else None
    return current_heading_text, path


def _char_offset_to_line(char_offset: int | None, line_offsets: list[int] | None) -> int | None:
    """Convert a character offset to a line number (1-indexed)."""
    if char_offset is None or not line_offsets:
        return None

    left, right = 0, len(line_offsets) - 1
    while left < right:
        mid = (left + right + 1) // 2
        if line_offsets[mid] <= char_offset:
            left = mid
        else:
            right = mid - 1

    return left + 1


def _get_cached_model_path(cache_dir: Path, model_name: str) -> Path | None:
    """Get the cached model directory path using huggingface_hub's snapshot_download."""
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
                    cache_dir=str(cache_dir_abs)
                )
                return Path(model_dir).resolve()
    except (ImportError, Exception):
        pass
    return None


def _ensure_embed_model(config: dict[str, Any]) -> None:
    """Ensure the embedding model is initialized."""
    global _embed_model_initialized

    if _embed_model_initialized:
        return

    model_name = config["retrieval"]["embed_model_name"]
    cache_dir = Path(config["storage"]["model_cache_dir"])
    offline_mode = config["retrieval"]["offline"]

    cached_model_path = _get_cached_model_path(cache_dir, model_name)
    if cached_model_path and offline_mode:
        logger.info(f"Loading embedding model from cache: {cached_model_path}")
        embed_model = FastEmbedEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir),
            specific_model_path=str(cached_model_path)
        )
    else:
        embed_model = FastEmbedEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir),
        )
    logger.info("Embedding model initialized successfully")
    Settings.embed_model = embed_model
    _embed_model_initialized = True


def _ensure_reranker(config: dict[str, Any]):
    """Load the FlashRank reranking model."""
    global _reranker_model

    if _reranker_model is not None:
        return _reranker_model

    try:
        from flashrank import Ranker
    except ImportError as exc:
        raise ImportError(
            "flashrank is required for reranking. Install with: pip install chunksilo"
        ) from exc

    model_name = config["retrieval"]["rerank_model_name"]
    cache_dir = Path(config["storage"]["model_cache_dir"])
    offline_mode = config["retrieval"]["offline"]

    model_mapping = {
        "cross-encoder/ms-marco-MiniLM-L-6-v2": "ms-marco-MiniLM-L-12-v2",
        "ms-marco-MiniLM-L-6-v2": "ms-marco-MiniLM-L-12-v2",
    }
    if model_name in model_mapping:
        model_name = model_mapping[model_name]
    elif model_name.startswith("cross-encoder/"):
        base_name = model_name.replace("cross-encoder/", "")
        if "L-6" in base_name:
            model_name = base_name.replace("L-6", "L-12")
        else:
            model_name = base_name

    try:
        _reranker_model = Ranker(model_name=model_name, cache_dir=str(cache_dir))
    except Exception as exc:
        if offline_mode:
            raise FileNotFoundError(
                f"Rerank model '{model_name}' not available in cache directory {cache_dir}. "
                "Download it before running in offline mode."
            ) from exc
        raise

    logger.info(f"Rerank model '{model_name}' loaded successfully")
    return _reranker_model


def _ensure_bm25_retriever(config: dict[str, Any]):
    """Load the BM25 retriever for file name matching."""
    global _bm25_retriever_cache

    if _bm25_retriever_cache is not None:
        return _bm25_retriever_cache

    storage_dir = Path(config["storage"]["storage_dir"])
    bm25_index_dir = storage_dir / "bm25_index"

    if not bm25_index_dir.exists():
        logger.warning(f"BM25 index not found at {bm25_index_dir}. Run indexing to create it.")
        return None

    try:
        from llama_index.retrievers.bm25 import BM25Retriever
        logger.info(f"Loading BM25 index from {bm25_index_dir}")
        _bm25_retriever_cache = BM25Retriever.from_persist_dir(str(bm25_index_dir))
        logger.info("BM25 retriever loaded successfully")
        return _bm25_retriever_cache
    except Exception as e:
        logger.error(f"Failed to load BM25 retriever: {e}")
        return None


def _format_bm25_matches(bm25_nodes: list[NodeWithScore], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Format BM25 file name matches for the response."""
    matched_files = []
    for node in bm25_nodes:
        if node.score is None or node.score <= 0:
            continue
        metadata = node.node.metadata or {}
        file_path = metadata.get("file_path", "")
        source_uri = _resolve_file_uri(file_path, config) if file_path else None
        matched_files.append({
            "uri": source_uri,
            "score": round(float(node.score), 4),
        })
    return matched_files[:5]


def _preprocess_query(query: str) -> str:
    """Preprocess queries with basic normalization."""
    if not query or not query.strip():
        return query

    original_query = query
    query = " ".join(query.split())
    query = query.rstrip(".,!?;")
    processed = query.strip()
    return processed if processed else original_query


def _prepare_confluence_query_terms(query: str) -> list[str]:
    """Prepare query terms for Confluence CQL search."""
    words = query.strip().lower().split()
    meaningful = [w for w in words if w not in CONFLUENCE_STOPWORDS and len(w) >= 2]
    return [w.replace('"', '\\"') for w in meaningful]


def _get_confluence_page_dates(
    base_url: str, page_id: str, username: str, api_token: str,
    ca_bundle_path: str | None = None
) -> dict[str, str]:
    """Fetch creation and modification dates for a Confluence page."""
    if requests is None:
        return {}

    try:
        url = f"{base_url.rstrip('/')}/wiki/api/v2/pages/{page_id}"
        response = requests.get(
            url,
            auth=(username, api_token),
            timeout=5.0,
            verify=ca_bundle_path if ca_bundle_path else True,
        )
        if response.status_code == 200:
            data = response.json()
            result = {}
            if "createdAt" in data:
                creation_date = _parse_iso8601_to_date(data["createdAt"])
                if creation_date:
                    result["creation_date"] = creation_date
            if "version" in data and "createdAt" in data["version"]:
                last_modified = _parse_iso8601_to_date(data["version"]["createdAt"])
                if last_modified:
                    result["last_modified_date"] = last_modified
            return result
    except Exception as e:
        logger.debug(f"Failed to fetch Confluence page dates: {e}")
    return {}


def _prepare_jira_jql_query(query: str, config: dict[str, Any]) -> str:
    """Construct a JQL query from user search terms and configuration.

    Uses Jira's 'text' field which searches across Summary, Description,
    Environment, Comments, and all text custom fields. This provides broad
    coverage similar to natural language search.

    Note: Fuzzy search operators (~) are deprecated in Jira Cloud but work
    in Data Center/Server. ChunkSilo's semantic search (embeddings + reranker)
    provides fuzzy matching regardless of Jira version.

    Args:
        query: User's search query string
        config: Configuration dict containing jira settings

    Returns:
        JQL query string ready for Jira API

    Raises:
        None - returns safe default query on edge cases

    Performance Note:
        For high-volume instances, consider using specific fields like
        "Summary ~ 'term' OR Description ~ 'term'" instead of "text ~ 'term'"
        to reduce search scope. The current implementation prioritizes recall.

    References:
        - Jira text field: https://support.atlassian.com/jira-software-cloud/docs/search-for-work-items-using-the-text-field/
        - JQL operators: https://support.atlassian.com/jira-software-cloud/docs/jql-operators/
    """
    # Reuse Confluence query term preparation for stopword filtering
    # This gives us a clean list of meaningful search terms
    query_terms = _prepare_confluence_query_terms(query)

    # Build the text search clause
    # Using JQL 'text' field which searches across all text fields for broad recall
    if not query_terms:
        # No meaningful terms after filtering, use original query
        escaped = query.strip().replace('"', '\\"')
        if not escaped:
            logger.warning("Jira search skipped: empty query after processing")
            return ""
        text_clause = f'text ~ "{escaped}"'
    elif len(query_terms) == 1:
        # Single term - simple text search
        text_clause = f'text ~ "{query_terms[0]}"'
    else:
        # Multiple terms - use OR logic to find issues matching any term
        # ChunkSilo's reranker will score results by relevance after retrieval
        text_conditions = ' OR '.join([f'text ~ "{term}"' for term in query_terms])
        text_clause = f'({text_conditions})'

    # Add project filter if configured
    # Empty projects list means search all accessible projects
    projects = config["jira"].get("projects", [])
    if projects:
        # Restrict search to specific project keys
        project_list = ", ".join([f'"{p}"' for p in projects])
        project_clause = f'project IN ({project_list})'
        jql = f'{text_clause} AND {project_clause}'
    else:
        jql = text_clause

    # Order by updated DESC for recency
    # This enables ChunkSilo's recency boost feature and returns most relevant recent issues first
    jql += ' ORDER BY updated DESC'

    return jql


def _jira_issue_to_text(issue, include_comments: bool, include_custom_fields: bool) -> str:
    """Convert a Jira issue to searchable text representation.

    This function constructs a text representation of a Jira issue that will
    be embedded and indexed by ChunkSilo's vector database. The text includes
    structured sections for issue metadata, description, comments, and custom
    fields, making it suitable for semantic search.

    Args:
        issue: JIRA issue object from the jira library
        include_comments: If True, include all issue comments with author names
        include_custom_fields: If True, include all custom field values

    Returns:
        Formatted text string suitable for embedding and search

    Format:
        Issue: PROJ-123
        Summary: Issue title here

        Description:
        Issue description text...

        Comments:
        - John Doe: Comment text
        - Jane Smith: Another comment

        Custom Fields:
        customfield_10001: value

    Note:
        Custom fields are detected by checking for attributes starting with
        'customfield_'. Only fields with non-empty values are included.

    Error Handling:
        Missing or None fields (description, comments) are gracefully skipped.
        The function always returns valid text even with minimal issue data.
    """
    parts = []

    # Issue key and summary are always included
    # These are the most important fields for identifying and understanding the issue
    parts.append(f"Issue: {issue.key}")
    parts.append(f"Summary: {issue.fields.summary}")

    # Description provides detailed context
    # Use hasattr() for safe field access since description can be missing/None
    if hasattr(issue.fields, 'description') and issue.fields.description:
        parts.append(f"\nDescription:\n{issue.fields.description}")

    # Comments provide discussion context and additional searchable content
    # Include author names for context (who said what matters for search)
    if include_comments and hasattr(issue.fields, 'comment'):
        comments = issue.fields.comment.comments
        if comments:
            parts.append("\nComments:")
            for comment in comments:
                # Safely extract author display name
                author = getattr(comment, 'author', None)
                author_name = author.displayName if author and hasattr(author, 'displayName') else 'Unknown'
                parts.append(f"- {author_name}: {comment.body}")

    # Custom fields provide instance-specific metadata
    # Detect by 'customfield_' prefix, include any with non-empty values
    if include_custom_fields:
        custom_fields = []
        for field_name in dir(issue.fields):
            if field_name.startswith('customfield_'):
                field_value = getattr(issue.fields, field_name, None)
                # Only include fields with meaningful values
                if field_value is not None and str(field_value).strip():
                    custom_fields.append(f"{field_name}: {field_value}")

        if custom_fields:
            parts.append("\nCustom Fields:")
            parts.extend(custom_fields)

    # Join all sections with newlines for structured, readable text
    # This format works well for semantic search and LLM processing
    return "\n".join(parts)


def _jira_issue_to_metadata(issue, jira_url: str) -> dict[str, Any]:
    """Extract structured metadata from a Jira issue.

    This function extracts metadata that will be attached to the search result
    node. The metadata is used for:
    - Display in search results (title, status, etc.)
    - Date range filtering (creation_date, last_modified_date)
    - Recency boosting (last_modified_date)
    - URI construction (issue_key)
    - Result grouping and sorting

    Args:
        issue: JIRA issue object from the jira library
        jira_url: Base Jira URL for constructing attachment URLs

    Returns:
        Dictionary of metadata fields following ChunkSilo conventions

    Metadata Fields:
        Required:
            - source: Always "Jira"
            - issue_key: Jira issue key (e.g., "PROJ-123")
            - issue_type: Type of issue ("Bug", "Story", etc.)
            - status: Current status ("Open", "In Progress", etc.)
            - title: Issue summary
            - file_name: Display name (format: "{key}: {summary}")

        Optional (present if available):
            - priority: Issue priority ("High", "Medium", "Low")
            - creation_date: ISO format date string (YYYY-MM-DD)
            - last_modified_date: ISO format date string (YYYY-MM-DD)
            - project_key: Project key (e.g., "PROJ")
            - project_name: Full project name
            - assignee: Assigned user's display name
            - reporter: Reporter's display name
            - attachments: List of attachment metadata (not content)

    Date Format:
        Jira returns ISO 8601 dates like "2024-01-15T10:30:00.000+0000".
        We convert to "YYYY-MM-DD" format for consistency with ChunkSilo's
        date filtering and display logic.

    Attachment Handling:
        Per requirements, we list attachment metadata but do NOT download
        or index attachment content. Users/models can follow URLs to access
        attachments if needed.

    Error Handling:
        Missing optional fields (priority, assignee, etc.) are gracefully
        skipped. The function always returns a valid metadata dict with
        at least the required fields.
    """
    # Required fields - always present
    metadata = {
        "source": "Jira",  # Identifies source for URI resolution and display
        "issue_key": issue.key,  # Unique identifier for linking
        "issue_type": issue.fields.issuetype.name,
        "status": issue.fields.status.name,
        "title": issue.fields.summary,
        # file_name format matches ChunkSilo convention for display
        "file_name": f"{issue.key}: {issue.fields.summary}",
    }

    # Optional: Priority (may not exist on all issue types)
    if hasattr(issue.fields, 'priority') and issue.fields.priority:
        metadata["priority"] = issue.fields.priority.name

    # Dates for filtering and recency boosting
    # Parse ISO 8601 format and convert to YYYY-MM-DD for consistency
    if hasattr(issue.fields, 'created') and issue.fields.created:
        creation_date = _parse_iso8601_to_date(issue.fields.created)
        if creation_date:
            metadata["creation_date"] = creation_date

    if hasattr(issue.fields, 'updated') and issue.fields.updated:
        # last_modified_date used for recency boost calculation
        last_modified = _parse_iso8601_to_date(issue.fields.updated)
        if last_modified:
            metadata["last_modified_date"] = last_modified

    # Project information
    if hasattr(issue.fields, 'project') and issue.fields.project:
        metadata["project_key"] = issue.fields.project.key
        metadata["project_name"] = issue.fields.project.name

    # People - use display names for human-readable output
    if hasattr(issue.fields, 'assignee') and issue.fields.assignee:
        metadata["assignee"] = issue.fields.assignee.displayName

    if hasattr(issue.fields, 'reporter') and issue.fields.reporter:
        metadata["reporter"] = issue.fields.reporter.displayName

    # Attachments - list metadata only, don't index content
    # Per requirements: provide URIs so user/model can access if needed
    if hasattr(issue.fields, 'attachment') and issue.fields.attachment:
        attachments = []
        for att in issue.fields.attachment:
            attachments.append({
                "filename": att.filename,
                "url": att.content,  # Direct download URL
                "size": att.size,  # Size in bytes
            })
        if attachments:
            metadata["attachments"] = attachments

    return metadata


def _search_confluence(query: str, config: dict[str, Any]) -> list[NodeWithScore]:
    """Search Confluence for documents matching the query using CQL."""
    base_url = config["confluence"]["url"]
    if not base_url:
        logger.warning("Confluence search skipped: confluence.url not set in config")
        return []

    if ConfluenceReader is None:
        logger.warning("llama-index-readers-confluence not installed, skipping Confluence search")
        return []

    username = config["confluence"]["username"]
    api_token = config["confluence"]["api_token"]
    max_results = config["confluence"]["max_results"]
    ca_bundle_path = config["ssl"]["ca_bundle_path"] or None

    if not (base_url and username and api_token):
        missing = []
        if not username:
            missing.append("confluence.username")
        if not api_token:
            missing.append("confluence.api_token")
        logger.warning(f"Confluence search skipped: missing {', '.join(missing)} in config")
        return []

    try:
        reader = ConfluenceReader(base_url=base_url, user_name=username, api_token=api_token)
        query_terms = _prepare_confluence_query_terms(query)

        if not query_terms:
            escaped = query.strip().replace('"', '\\"')
            if not escaped:
                logger.warning("Confluence search skipped: empty query after processing")
                return []
            cql = f'text ~ "{escaped}" AND type = "page"'
        elif len(query_terms) == 1:
            cql = f'text ~ "{query_terms[0]}" AND type = "page"'
        else:
            text_conditions = ' OR '.join([f'text ~ "{term}"' for term in query_terms])
            cql = f'({text_conditions}) AND type = "page"'

        documents = reader.load_data(cql=cql, max_num_results=max_results)

        nodes: list[NodeWithScore] = []
        for doc in documents:
            metadata = doc.metadata.copy()
            metadata["source"] = "Confluence"
            if "title" in metadata:
                metadata["file_name"] = metadata["title"]

            page_id = metadata.get("page_id")
            if page_id:
                date_info = _get_confluence_page_dates(base_url, page_id, username, api_token, ca_bundle_path)
                metadata.update(date_info)

            node = TextNode(text=doc.text, metadata=metadata)
            nodes.append(NodeWithScore(node=node, score=0.0))

        return nodes

    except Exception as e:
        logger.error(f"Failed to search Confluence: {e}", exc_info=True)
        return []


def _search_jira(query: str, config: dict[str, Any]) -> list[NodeWithScore]:
    """Search Jira for issues matching the query using JQL.

    This function performs a real-time search against the configured Jira
    instance using JQL (Jira Query Language). Results are converted to
    ChunkSilo's NodeWithScore format and merged with other search results.

    The function follows ChunkSilo's integration pattern (same as Confluence):
    - Returns empty list on errors (graceful degradation)
    - Logs warnings for configuration issues
    - Uses timeout protection in calling code (ThreadPoolExecutor)
    - Converts results to standard NodeWithScore format

    Search Strategy:
        - Uses JQL 'text' field for broad search across all text fields
        - Relies on ChunkSilo's semantic search (embeddings + reranker) for
          relevance ranking instead of traditional fuzzy search
        - Returns results with score=0.0; FlashRank reranker scores them

    Args:
        query: User's search query string
        config: Configuration dictionary with jira and ssl settings

    Returns:
        List of NodeWithScore objects, or empty list on error/disabled

    Configuration Requirements:
        config["jira"]["url"]: Jira base URL (empty = disabled)
        config["jira"]["username"]: Jira username/email (required for Cloud, optional for Server PAT)
        config["jira"]["api_token"]: API token (Cloud) or Personal Access Token (Server/Data Center)
        config["jira"]["max_results"]: Maximum issues to return
        config["jira"]["projects"]: List of project keys (empty = all)
        config["jira"]["include_comments"]: Include issue comments
        config["jira"]["include_custom_fields"]: Include custom fields
        config["ssl"]["ca_bundle_path"]: Optional SSL CA bundle path

    Error Handling:
        - Empty URL: Returns [] with debug log
        - Missing library: Returns [] with warning log
        - Missing credentials: Returns [] with warning log
        - API errors: Returns [] with error log (exc_info=True)
        - All errors are non-fatal to allow other searches to succeed

    Performance:
        - Respects max_results limit (default 30)
        - Orders by updated DESC (most recent first)
        - Fetches all fields including custom fields

    SSL/TLS:
        - Supports custom CA bundles via ssl.ca_bundle_path
        - Automatically configured through jira_options["verify"]

    Authentication:
        - Jira Cloud: Set both username (email) and api_token (API token)
          Uses basic auth internally
        - Jira Server/Data Center with PAT: Set only api_token (Personal Access Token)
          Leave username empty; uses bearer token auth internally
        - Jira Server/Data Center with password: Set username and api_token (password)
          Uses basic auth internally (if basic auth enabled on server)

    References:
        - Jira REST API: https://developer.atlassian.com/cloud/jira/platform/rest/v3/
        - JQL Reference: https://support.atlassian.com/jira-software-cloud/docs/use-advanced-search-with-jira-query-language-jql/
    """
    # Check if Jira integration is enabled via URL
    base_url = config["jira"]["url"]
    if not base_url:
        logger.debug("Jira search skipped: jira.url not set in config")
        return []

    # Gracefully degrade if optional dependency not installed
    if JIRA is None:
        logger.warning("jira library not installed, skipping Jira search")
        return []

    # Extract configuration settings
    username = config["jira"]["username"]
    api_token = config["jira"]["api_token"]
    max_results = config["jira"]["max_results"]
    include_comments = config["jira"]["include_comments"]
    include_custom_fields = config["jira"]["include_custom_fields"]
    ca_bundle_path = config["ssl"]["ca_bundle_path"] or None

    # Validate required credentials are present
    # For Jira Cloud: both username and api_token required (basic auth)
    # For Jira Server/Data Center with PAT: only api_token required (token auth)
    if not api_token:
        logger.warning("Jira search skipped: missing jira.api_token in config")
        return []

    try:
        # Configure SSL certificate verification if CA bundle provided
        # This enables Jira integration in corporate environments with custom CAs
        jira_options = {"server": base_url}
        if ca_bundle_path:
            jira_options["verify"] = ca_bundle_path

        # Choose authentication method based on credentials provided:
        # - Username + API token: Use basic auth (Jira Cloud, or Server with password)
        # - API token only: Use token auth (Jira Server/Data Center with PAT)
        if username:
            # Basic auth for Jira Cloud (username + API token)
            # Also works for Jira Server with username + password
            jira_client = JIRA(
                options=jira_options,
                basic_auth=(username, api_token)
            )
        else:
            # Token auth for Jira Server/Data Center Personal Access Tokens (PAT)
            # PATs are used alone without a username
            jira_client = JIRA(
                options=jira_options,
                token_auth=api_token
            )

        # Construct JQL with text search and project filtering
        jql = _prepare_jira_jql_query(query, config)
        if not jql:
            # Empty query after processing
            return []

        # Log JQL query for debugging
        logger.debug(f"Jira JQL query: {jql}")

        # Fetch all fields including custom fields for comprehensive search
        # maxResults limits API response size for performance
        issues = jira_client.search_issues(
            jql,
            maxResults=max_results,
            fields='*all'  # Get all fields including custom fields
        )

        # Convert issues to NodeWithScore format
        nodes: list[NodeWithScore] = []
        for issue in issues:
            # Build searchable text representation
            # This will be embedded and indexed by ChunkSilo's vector database
            text = _jira_issue_to_text(issue, include_comments, include_custom_fields)

            # Extract structured metadata for filtering and display
            metadata = _jira_issue_to_metadata(issue, base_url)

            # Create node with text and metadata
            # Initial score is 0.0; FlashRank reranker will assign relevance scores
            node = TextNode(text=text, metadata=metadata)
            nodes.append(NodeWithScore(node=node, score=0.0))

        logger.debug(f"Jira search returned {len(nodes)} issues")
        # Return results for merging with other search sources
        return nodes

    except Exception as e:
        # Catch all exceptions to prevent search pipeline failure
        # Log errors with full traceback for debugging
        logger.error(f"Failed to search Jira: {e}", exc_info=True)
        # Return empty list to allow search to continue with other sources
        return []


def load_llamaindex_index(config: dict[str, Any] | None = None):
    """Load the LlamaIndex from storage."""
    if config is None:
        config = _get_config()
    global _index_cache

    if _index_cache is not None:
        return _index_cache

    storage_dir = Path(config["storage"]["storage_dir"])
    if not storage_dir.exists():
        raise FileNotFoundError(
            f"Storage directory {storage_dir} does not exist. "
            "Please run indexing first."
        )

    logger.info("Loading LlamaIndex from storage...")
    _ensure_embed_model(config)
    storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
    index = load_index_from_storage(storage_context)
    _index_cache = index
    return index


def _parse_iso8601_to_date(iso_string: str) -> str | None:
    """Parse ISO 8601 timestamp to YYYY-MM-DD format.

    Handles various ISO 8601 formats including:
    - Z suffix: 2024-01-15T10:30:00Z
    - Timezone with colon: 2024-01-15T10:30:00.000+00:00
    - Timezone without colon: 2024-01-15T10:30:00.000+0000 (Jira format)

    Args:
        iso_string: ISO 8601 formatted datetime string

    Returns:
        Date in YYYY-MM-DD format, or None if parsing fails
    """
    if not iso_string:
        return None

    try:
        # Normalize the timestamp
        normalized = iso_string.strip()

        # Replace Z suffix with +00:00
        normalized = normalized.replace('Z', '+00:00')

        # Insert colon in timezone offsets like +0000 → +00:00
        # Matches ±HHMM at end of string, inserts colon: ±HH:MM
        normalized = re.sub(r'([+-]\d{2})(\d{2})$', r'\1:\2', normalized)

        # Parse and format
        dt = datetime.fromisoformat(normalized)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


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
    """Filter nodes by date range."""
    if not date_from and not date_to:
        return nodes

    from_dt = _parse_date(date_from) if date_from else None
    to_dt = _parse_date(date_to) if date_to else None

    filtered = []
    for node in nodes:
        metadata = node.node.metadata or {}
        doc_date_str = metadata.get("last_modified_date") or metadata.get("creation_date")
        if not doc_date_str:
            filtered.append(node)
            continue

        doc_date = _parse_date(doc_date_str)
        if not doc_date:
            filtered.append(node)
            continue

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
    """Apply time-decay boost to nodes based on document recency."""
    if not nodes or boost_weight <= 0:
        return nodes

    today = datetime.now()
    boosted_nodes = []

    for node in nodes:
        metadata = node.node.metadata or {}
        doc_date_str = metadata.get("last_modified_date") or metadata.get("creation_date")
        base_score = node.score if node.score is not None else 0.5

        if not doc_date_str:
            boosted_nodes.append(NodeWithScore(node=node.node, score=base_score))
            continue

        doc_date = _parse_date(doc_date_str)
        if not doc_date:
            boosted_nodes.append(NodeWithScore(node=node.node, score=base_score))
            continue

        age_days = (today - doc_date).days
        if age_days < 0:
            age_days = 0

        decay_rate = math.log(2) / half_life_days
        recency_factor = math.exp(-decay_rate * age_days)
        boosted_score = base_score * (1 + boost_weight * recency_factor)

        boosted_nodes.append(NodeWithScore(node=node.node, score=boosted_score))

    boosted_nodes.sort(key=lambda x: x.score or 0, reverse=True)
    return boosted_nodes


def run_search(
    query: str,
    date_from: str | None = None,
    date_to: str | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Execute the full search pipeline and return structured results.

    This is the shared implementation used by both the MCP tool and CLI.

    Args:
        query: Search query text
        date_from: Optional start date filter (YYYY-MM-DD, inclusive)
        date_to: Optional end date filter (YYYY-MM-DD, inclusive)
        config_path: Optional path to config.yaml

    Returns:
        Structured response dict with matched_files, chunks, etc.
    """
    config = _init_config(config_path) if config_path else _get_config()

    # Setup environment on first call
    _setup_offline_mode(config)
    _setup_ssl(config)

    start_time = time.time()

    try:
        enhanced_query = _preprocess_query(query)

        # Load index
        index = load_llamaindex_index(config)

        # Stage 1a: Vector search
        embed_top_k = config["retrieval"]["embed_top_k"]
        retriever = index.as_retriever(similarity_top_k=embed_top_k)
        vector_nodes = retriever.retrieve(enhanced_query)

        # Stage 1b: BM25 file name search
        matched_files: list[dict[str, Any]] = []
        bm25_retriever = _ensure_bm25_retriever(config)
        if bm25_retriever:
            try:
                bm25_matches = bm25_retriever.retrieve(enhanced_query)
                if bm25_matches:
                    matched_files = _format_bm25_matches(bm25_matches, config)
                    logger.info(f"BM25 matched {len(matched_files)} files (from {len(bm25_matches)} candidates)")
            except Exception as e:
                logger.error(f"BM25 search failed: {e}")

        nodes = vector_nodes

        # Search Confluence (with timeout)
        confluence_nodes: list[NodeWithScore] = []
        confluence_timeout = config["confluence"]["timeout"]
        if config["confluence"]["url"]:
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_search_confluence, enhanced_query, config)
                    confluence_nodes = future.result(timeout=confluence_timeout)
                logger.info(f"Confluence search returned {len(confluence_nodes)} entries")
            except FuturesTimeoutError:
                logger.warning(f"Confluence search timed out after {confluence_timeout}s")
            except Exception as e:
                logger.error(f"Error during Confluence search: {e}")

        if confluence_nodes:
            nodes.extend(confluence_nodes)

        # Search Jira (with timeout)
        jira_nodes: list[NodeWithScore] = []
        jira_timeout = config["jira"]["timeout"]
        if config["jira"]["url"]:
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_search_jira, enhanced_query, config)
                    jira_nodes = future.result(timeout=jira_timeout)
                logger.info(f"Jira search returned {len(jira_nodes)} entries")
            except FuturesTimeoutError:
                logger.warning(f"Jira search timed out after {jira_timeout}s")
            except Exception as e:
                logger.error(f"Error during Jira search: {e}")

        if jira_nodes:
            nodes.extend(jira_nodes)

        # Apply date filtering
        if date_from or date_to:
            original_count = len(nodes)
            nodes = _filter_nodes_by_date(nodes, date_from, date_to)
            logger.info(f"Date filtering: {original_count} -> {len(nodes)} nodes")

        # Apply recency boost
        recency_boost = config["retrieval"]["recency_boost"]
        recency_half_life = config["retrieval"]["recency_half_life_days"]
        if recency_boost > 0:
            nodes = _apply_recency_boost(nodes, recency_boost, recency_half_life)

        # Cap candidates before reranking
        rerank_candidates = config["retrieval"]["rerank_candidates"]
        if len(nodes) > rerank_candidates:
            logger.info(f"Capping rerank candidates: {len(nodes)} -> {rerank_candidates}")
            nodes = nodes[:rerank_candidates]

        # Stage 2: Rerank
        rerank_top_k = config["retrieval"]["rerank_top_k"]
        rerank_scores: dict[int, float] = {}
        if nodes:
            rerank_limit = max(1, min(rerank_top_k, len(nodes)))
            try:
                reranker = _ensure_reranker(config)
                passages = [{"text": node.node.get_content() or ""} for node in nodes]

                from flashrank import RerankRequest
                rerank_request = RerankRequest(query=enhanced_query, passages=passages)
                reranked_results = reranker.rerank(rerank_request)

                text_to_indices: dict[str, list[tuple[int, NodeWithScore]]] = {}
                for idx, node in enumerate(nodes):
                    node_text = node.node.get_content() or ""
                    if node_text not in text_to_indices:
                        text_to_indices[node_text] = []
                    text_to_indices[node_text].append((idx, node))

                reranked_nodes = []
                seen_indices: set[int] = set()
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

                for idx, node in enumerate(nodes):
                    if idx not in seen_indices:
                        reranked_nodes.append(node)

                nodes = reranked_nodes[:rerank_limit]
            except Exception as e:
                logger.error(f"Reranking failed, falling back to vector search order: {e}")
                nodes = nodes[:rerank_limit]

        # Filter by score threshold
        score_threshold = config["retrieval"]["score_threshold"]
        if score_threshold > 0:
            nodes = [
                node for node in nodes
                if rerank_scores.get(id(node), 0.0) >= score_threshold
            ]

        # Format chunks
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

            # Build heading path
            headings = metadata.get("document_headings") or metadata.get("headings") or []
            if not headings and file_path:
                headings = get_heading_store().get_headings(str(file_path))
            char_start = getattr(node.node, "start_char_idx", None)
            heading_text = metadata.get("heading")
            heading_path: list[str] = []
            if isinstance(headings, list) and headings:
                if heading_text is None and char_start is not None:
                    heading_text, heading_path = _build_heading_path(headings, char_start)
            meta_heading_path = metadata.get("heading_path")
            if not heading_path and meta_heading_path:
                heading_path = list(meta_heading_path)
            if heading_text and (not heading_path or heading_path[-1] != heading_text):
                heading_path = heading_path + [heading_text] if heading_path else [heading_text]

            # Build URI
            source_uri = None
            if original_source == "Confluence":
                confluence_url = config["confluence"]["url"]
                page_id = metadata.get("page_id")
                if confluence_url and page_id:
                    source_uri = f"{confluence_url.rstrip('/')}/pages/viewpage.action?pageId={page_id}"
                elif confluence_url:
                    title = metadata.get("title", metadata.get("file_name", ""))
                    if title:
                        from urllib.parse import quote
                        encoded_title = quote(title.replace(" ", "+"))
                        source_uri = f"{confluence_url.rstrip('/')}/spaces/~{encoded_title}"
            elif original_source == "Jira":
                # Build Jira issue URI using standard browse URL format
                jira_url = config["jira"]["url"]
                issue_key = metadata.get("issue_key")
                if jira_url and issue_key:
                    source_uri = f"{jira_url.rstrip('/')}/browse/{issue_key}"
            elif file_path:
                source_uri = _resolve_file_uri(file_path, config)

            page_number = (
                metadata.get("page_label")
                or metadata.get("page_number")
                or metadata.get("page")
            )

            line_number = None
            line_offsets = metadata.get("line_offsets")
            if line_offsets and char_start is not None:
                line_number = _char_offset_to_line(char_start, line_offsets)

            location = {
                "uri": source_uri,
                "page": page_number,
                "line": line_number,
                "heading_path": heading_path if heading_path else None,
            }

            score_value = rerank_scores.get(id(node), getattr(node, "score", None))
            chunk_data = {
                "text": chunk_text,
                "score": round(float(score_value), 3) if score_value is not None else 0.0,
                "location": location,
            }
            chunks.append(chunk_data)

        elapsed = time.time() - start_time

        return {
            "matched_files": matched_files,
            "num_matched_files": len(matched_files),
            "chunks": chunks,
            "num_chunks": len(chunks),
            "query": query,
            "retrieval_time": f"{elapsed:.2f}s",
        }

    except Exception as e:
        logger.error(f"Error in search: {e}", exc_info=True)
        return {
            "matched_files": [],
            "chunks": [],
            "error": str(e),
            "query": query,
        }
