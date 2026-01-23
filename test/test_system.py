#!/usr/bin/env python3
"""Pytest-based test script for the RAG system."""
import sys
from pathlib import Path
import pytest
import asyncio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from index import load_index_config, build_index
from chunksilo import STORAGE_DIR, load_llamaindex_index



def test_ingestion():
    """Test the ingestion pipeline."""
    print("=" * 60)
    print("Testing Ingestion Pipeline")
    print("=" * 60)

    try:
        config = load_index_config()
        if not config.directories:
            pytest.skip("No directories configured in config.yaml")
    except FileNotFoundError:
        pytest.skip("config.yaml not found; create config before running ingestion tests.")

    build_index()
    print("✓ Ingestion completed successfully")


def test_query():
    """Test the retrieval functionality (no LLM inside the MCP server)."""
    print("\n" + "=" * 60)
    print("Testing Query Functionality")
    print("=" * 60)

    # Test queries
    test_queries = [
        "What is this document about?",
        "Summarize the main topics",
        "What are the key points?",
    ]

    from chunksilo import search_docs

    async def _run_queries():
        if not (STORAGE_DIR / "docstore.json").exists():
            pytest.skip("Index not built (docstore.json missing); run ingestion before query tests.")

        # Load index to verify it exists
        index = load_llamaindex_index()
        print("✓ Index loaded successfully")

        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 60)

            result = await search_docs(query)
            chunks = result.get("chunks", [])
            print(f"Retrieved {len(chunks)} chunks")

            if chunks:
                top = chunks[0]
                print(f"Top chunk score: {top.get('score', 'N/A')}")
                print(f"Top chunk preview: {top.get('text', '')[:200]}...")

            if "retrieval_time" in result:
                print(f"Retrieval time: {result['retrieval_time']}")


    asyncio.run(_run_queries())


