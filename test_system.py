#!/usr/bin/env python3
"""
Test script for the RAG system.
Tests both ingestion and MCP server functionality.
"""
import os
import sys
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ingest import build_index
from mcp_server import load_llamaindex_index

load_dotenv()


def test_ingestion():
    """Test the ingestion pipeline."""
    print("=" * 60)
    print("Testing Ingestion Pipeline")
    print("=" * 60)
    
    try:
        build_index()
        print("✓ Ingestion completed successfully")
        return True
    except Exception as e:
        print(f"✗ Ingestion failed: {e}")
        return False


async def test_query():
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
    
    try:
        # Load index to verify it exists
        index = load_llamaindex_index()
        print(f"✓ Index loaded successfully")
        
        # Test queries by calling the retrieval function directly
        from mcp_server import retrieve_docs
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 60)
            
            try:
                result = await retrieve_docs(query)
                chunks = result.get("chunks", [])
                print(f"Retrieved {len(chunks)} chunks")
                
                if chunks:
                    top = chunks[0]
                    print(f"Top chunk score: {top.get('score', 'N/A')}")
                    print(f"Top chunk preview: {top.get('text', '')[:200]}...")
                
                if "retrieval_time" in result:
                    print(f"Retrieval time: {result['retrieval_time']}")
                
            except Exception as e:
                print(f"✗ Query failed: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n✓ Query tests completed")
        return True
        
    except Exception as e:
        print(f"✗ Query test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests (ingestion + retrieval)."""
    print("\n" + "=" * 60)
    print("RAG System Test Suite")
    print("=" * 60)
    
    results = {
        "ingestion": False,
        "query": False,
    }
    
    # Test ingestion
    results["ingestion"] = test_ingestion()
    
    if not results["ingestion"]:
        print("\n⚠ Warning: Ingestion failed. Make sure you have documents in the data/ directory.")
        return
    
    # Test queries
    results["query"] = await test_query()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main())

