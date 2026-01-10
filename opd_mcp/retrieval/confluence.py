"""Confluence search integration."""

import logging
import os
from datetime import datetime
from typing import List, Optional

from llama_index.core.schema import TextNode, NodeWithScore

from opd_mcp.config import (
    CA_BUNDLE_PATH,
    CONFLUENCE_MAX_RESULTS,
    CONFLUENCE_STOPWORDS,
)

logger = logging.getLogger(__name__)

# Optional imports for Confluence
try:
    from llama_index.readers.confluence import ConfluenceReader
    import requests
except ImportError:
    ConfluenceReader = None
    requests = None


def prepare_confluence_query_terms(query: str) -> List[str]:
    """Prepare query terms for Confluence CQL search.

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


def get_confluence_page_dates(
    base_url: str, page_id: str, username: str, api_token: str
) -> dict:
    """Fetch creation and modification dates for a Confluence page.

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
                try:
                    dt = datetime.fromisoformat(data["createdAt"].replace("Z", "+00:00"))
                    result["creation_date"] = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
            if "version" in data and "createdAt" in data["version"]:
                try:
                    dt = datetime.fromisoformat(
                        data["version"]["createdAt"].replace("Z", "+00:00")
                    )
                    result["last_modified_date"] = dt.strftime("%Y-%m-%d")
                except Exception:
                    pass
            return result
    except Exception as e:
        logger.debug(f"Failed to fetch Confluence page dates for {page_id}: {e}")
    return {}


def search_confluence(query: str) -> List[NodeWithScore]:
    """Search Confluence for documents matching the query using CQL.

    Uses OR logic for multi-word queries to cast a wider net, relying on the
    FlashRank reranker to identify the most semantically relevant results.
    Filters out common stopwords to improve search precision.

    Args:
        query: Search query string

    Returns:
        List of NodeWithScore objects compatible with the reranker
    """
    base_url = os.getenv("CONFLUENCE_URL")
    if not base_url:
        logger.warning("Confluence search skipped: CONFLUENCE_URL not set")
        return []

    if ConfluenceReader is None:
        logger.warning(
            "llama-index-readers-confluence not installed, skipping Confluence search"
        )
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
        query_terms = prepare_confluence_query_terms(query)

        # Build CQL query using OR logic to cast a wider net
        if not query_terms:
            # All words were stopwords - fall back to using original query as phrase
            escaped = query.strip().replace('"', '\\"')
            if not escaped:
                logger.warning("Confluence search skipped: empty query after processing")
                return []
            cql = f'text ~ "{escaped}" AND type = "page"'
        elif len(query_terms) == 1:
            cql = f'text ~ "{query_terms[0]}" AND type = "page"'
        else:
            # Multiple words: use OR logic to find pages with ANY matching word
            text_conditions = " OR ".join([f'text ~ "{term}"' for term in query_terms])
            cql = f'({text_conditions}) AND type = "page"'

        logger.debug(f"Confluence CQL query: {cql}")
        documents = reader.load_data(cql=cql, max_num_results=CONFLUENCE_MAX_RESULTS)

        nodes: List[NodeWithScore] = []
        for doc in documents:
            # Create a TextNode from the Confluence document
            metadata = doc.metadata.copy()
            metadata["source"] = "Confluence"
            if "title" in metadata:
                metadata["file_name"] = metadata["title"]

            # Fetch dates for this page
            page_id = metadata.get("page_id")
            if page_id:
                date_info = get_confluence_page_dates(base_url, page_id, username, api_token)
                metadata.update(date_info)

            node = TextNode(text=doc.text, metadata=metadata)
            nodes.append(NodeWithScore(node=node, score=0.0))

        return nodes

    except Exception as e:
        logger.error(f"Failed to search Confluence: {e}", exc_info=True)
        return []
