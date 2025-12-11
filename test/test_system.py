#!/usr/bin/env python3
"""Pytest-based test script for the RAG system."""
import sys
from pathlib import Path
import pytest
import asyncio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest import DATA_DIR, build_index
from mcp_server import STORAGE_DIR, load_llamaindex_index



def test_ingestion():
    """Test the ingestion pipeline."""
    print("=" * 60)
    print("Testing Ingestion Pipeline")
    print("=" * 60)

    if not DATA_DIR.exists():
        pytest.skip("DATA_DIR is missing; add documents before running ingestion tests.")

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

    from mcp_server import retrieve_docs

    async def _run_queries():
        if not STORAGE_DIR.exists():
            pytest.skip("STORAGE_DIR is missing; run ingestion before query tests.")

        # Load index to verify it exists
        index = load_llamaindex_index()
        print("✓ Index loaded successfully")

        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 60)

            result = await retrieve_docs(query)
            chunks = result.get("chunks", [])
            print(f"Retrieved {len(chunks)} chunks")

            if chunks:
                top = chunks[0]
                print(f"Top chunk score: {top.get('score', 'N/A')}")
                print(f"Top chunk preview: {top.get('text', '')[:200]}...")

            if "retrieval_time" in result:
                print(f"Retrieval time: {result['retrieval_time']}")


    asyncio.run(_run_queries())


def test_startup_log_buffer():
    """Test the startup log buffer is populated."""
    print("\n" + "=" * 60)
    print("Testing Startup Log Buffer")
    print("=" * 60)
    
    from mcp_server import _startup_log_buffer
    
    assert len(_startup_log_buffer) > 0, "Startup log buffer should be populated"
    assert any("Storage Directory" in msg for msg in _startup_log_buffer), "Should contain storage info"
    assert any("Embedding Model" in msg for msg in _startup_log_buffer), "Should contain model info"
    
    print("Buffered startup logs:")
    for msg in _startup_log_buffer:
        print(f"  {msg}")
    print("✓ Startup log buffer is properly populated")
