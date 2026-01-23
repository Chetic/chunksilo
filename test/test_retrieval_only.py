#!/usr/bin/env python3
"""Test the RAG system in retrieval-only mode (no LLM in the MCP server)."""
import sys
import traceback
from pathlib import Path
from dotenv import load_dotenv
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from index import load_index_config, build_index
from chunksilo import STORAGE_DIR, load_llamaindex_index

load_dotenv()


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


def test_index_loading():
    """Test loading the index."""
    print("\n" + "=" * 60)
    print("Testing Index Loading")
    print("=" * 60)

    if not STORAGE_DIR.exists():
        pytest.skip("STORAGE_DIR is missing; run ingestion before index loading tests.")

    index = load_llamaindex_index()
    print("✓ Index loaded successfully")

    # Test retrieval (without generation - no LLM needed)
    print("\nTesting retrieval (without LLM generation)...")
    retriever = index.as_retriever(similarity_top_k=3)
    query = "What is this document about?"
    nodes = retriever.retrieve(query)

    print(f"✓ Retrieved {len(nodes)} relevant chunks")
    if nodes:
        print(f"  Top chunk preview: {nodes[0].node.get_content()[:100]}...")
        if len(nodes) > 1:
            print(f"  Second chunk preview: {nodes[1].node.get_content()[:100]}...")



def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RAG System Test (Retrieval Only)")
    print("=" * 60)
    
    results = {
        "ingestion": False,
        "index_loading": False,
    }
    
    # Test ingestion
    for name, fn in ("ingestion", test_ingestion), ("index_loading", test_index_loading):
        try:
            fn()
            results[name] = True
        except Exception as e:
            print(f"✗ {name.replace('_', ' ').title()} failed: {e}")
            traceback.print_exc()
            results[name] = False

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
        print("\nYou can now start the MCP server and use it from an MCP-aware client (e.g., Continue).")
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()


