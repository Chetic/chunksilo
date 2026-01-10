"""Ranking and scoring utilities for retrieval."""

import math
from datetime import datetime
from typing import List, Optional

from llama_index.core.schema import NodeWithScore


def reciprocal_rank_fusion(
    results_lists: List[List[NodeWithScore]], k: int = 60
) -> List[NodeWithScore]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF Score = sum(1 / (k + rank)) for each list the document appears in.

    Args:
        results_lists: List of ranked result lists (each list is NodeWithScore)
        k: Constant to prevent high scores for top-ranked items (default 60)

    Returns:
        Combined and re-ranked list of NodeWithScore with RRF scores
    """
    rrf_scores: dict[str, float] = {}
    node_map: dict[str, NodeWithScore] = {}

    for results in results_lists:
        for rank, node in enumerate(results, start=1):
            node_id = node.node.node_id
            rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + 1.0 / (k + rank)
            if node_id not in node_map:
                node_map[node_id] = node

    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [NodeWithScore(node=node_map[nid].node, score=rrf_scores[nid]) for nid in sorted_ids]


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def filter_nodes_by_date(
    nodes: List[NodeWithScore],
    date_from: Optional[str],
    date_to: Optional[str],
) -> List[NodeWithScore]:
    """Filter nodes by date range.

    Args:
        nodes: List of nodes to filter
        date_from: Optional start date (YYYY-MM-DD, inclusive)
        date_to: Optional end date (YYYY-MM-DD, inclusive)

    Returns:
        Filtered list of nodes. Nodes without dates pass through for backward compatibility.
    """
    if not date_from and not date_to:
        return nodes

    from_dt = parse_date(date_from) if date_from else None
    to_dt = parse_date(date_to) if date_to else None

    filtered = []
    for node in nodes:
        metadata = node.node.metadata or {}
        # Check last_modified_date first, fall back to creation_date
        doc_date_str = metadata.get("last_modified_date") or metadata.get("creation_date")
        if not doc_date_str:
            # No date info - include by default (backward compatibility)
            filtered.append(node)
            continue

        doc_date = parse_date(doc_date_str)
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


def apply_recency_boost(
    nodes: List[NodeWithScore],
    boost_weight: float,
    half_life_days: int = 365,
) -> List[NodeWithScore]:
    """Apply time-decay boost to nodes based on document recency.

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

        doc_date = parse_date(doc_date_str)
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
