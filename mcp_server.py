#!/usr/bin/env python3
"""
MCP server for querying documentation using RAG.
Returns raw document chunks for the calling LLM to synthesize.
"""
import os
import time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load environment variables
load_dotenv()

# Configuration
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "./storage"))
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "5"))
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "BAAI/bge-small-en-v1.5")

# Initialize FastMCP server
mcp = FastMCP("llamaindex-docs-rag")

# Global caches
_index_cache = None
_embed_model_initialized = False


def _ensure_embed_model():
    """
    Ensure the same embedding model used during ingestion is available at query time.

    If this is not set, LlamaIndex falls back to its default (typically an OpenAI
    embedding model), which would require an OPENAI_API_KEY and cause failures
    inside the MCP server.
    """
    global _embed_model_initialized

    if _embed_model_initialized:
        return

    embed_model = HuggingFaceEmbedding(model_name=EMB_MODEL_NAME)
    Settings.embed_model = embed_model
    _embed_model_initialized = True


def load_llamaindex_index():
    """Load the LlamaIndex from storage."""
    global _index_cache

    if _index_cache is not None:
        return _index_cache

    if not STORAGE_DIR.exists():
        raise FileNotFoundError(
            f"Storage directory {STORAGE_DIR} does not exist. "
            "Please run ingest.py first."
        )

    # Make sure the embedding model is configured before using the index so that
    # query embeddings use the same model as ingestion (HuggingFace, not OpenAI).
    _ensure_embed_model()

    storage_context = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
    index = load_index_from_storage(storage_context)
    _index_cache = index
    return index


@mcp.tool()
async def retrieve_docs(query: str) -> dict[str, Any]:
    """
    Search the local PDF/DOCX/Markdown documentation corpus and return relevant chunks.
    
    This tool performs semantic search and returns raw document chunks.
    The calling LLM (e.g., Continue) will synthesize the answer from these chunks.
    
    Args:
        query: Natural language question or search query about the documentation
        
    Returns:
        Dictionary with 'chunks' (list of relevant document chunks) and metadata.
        Each chunk includes the full text, relevance score, and source metadata.
    """
    start_time = time.time()
    
    try:
        # Load index
        index = load_llamaindex_index()
        
        # Use retriever (no LLM needed - just retrieval)
        retriever = index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)
        
        # Retrieve relevant chunks
        nodes = retriever.retrieve(query)
        
        # Format chunks with full content and metadata
        chunks = []
        for node in nodes:
            chunk_data = {
                "text": node.node.get_content(),  # Full content, not truncated
                "score": float(node.score) if hasattr(node, 'score') and node.score is not None else 0.0,
                "metadata": node.node.metadata,
            }
            chunks.append(chunk_data)
        
        elapsed = time.time() - start_time
        
        return {
            "chunks": chunks,
            "query": query,
            "num_chunks": len(chunks),
            "retrieval_time": f"{elapsed:.2f}s",
        }
        
    except Exception as e:
        return {
            "chunks": [],
            "error": str(e),
            "query": query,
        }


if __name__ == "__main__":
    mcp.run()

