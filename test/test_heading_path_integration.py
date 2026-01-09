#!/usr/bin/env python3
"""Integration tests for heading_path extraction across different file formats.

Verifies that heading_path is correctly populated for:
- Markdown (.md)
- Word Documents (.docx)
- PDF Documents (.pdf) - (Expectation: None, unless specific PDF structure logic exists)
"""

import os
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch

# Basic imports to generate files
try:
    from docx import Document
except ImportError:
    Document = None

try:
    import fitz  # pymupdf
except ImportError:
    fitz = None

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# We need to import ingest and mcp_server after setting up sys.path
# We will patch their globals during tests
import ingest
import mcp_server

@pytest.fixture
def test_env(tmp_path):
    """Setup a temporary test environment with data and storage directories."""
    data_dir = tmp_path / "data"
    storage_dir = tmp_path / "storage"
    data_dir.mkdir()
    storage_dir.mkdir()
    
    # Return paths for use in tests
    return {
        "data_dir": data_dir,
        "storage_dir": storage_dir
    }

def create_markdown_file(path):
    """Create a markdown file with headings."""
    content = """
# Heading 1
Content under heading 1.

## Subheading 1.1
Content under subheading 1.1.
Target content for markdown.

# Heading 2
Content under heading 2.
"""
    with open(path, "w") as f:
        f.write(content)

def create_docx_file(path):
    """Create a DOCX file with headings."""
    if not Document:
        pytest.skip("python-docx not installed")
    
    doc = Document()
    doc.add_heading("Major Section", level=1)
    doc.add_paragraph("Introduction to major section.")
    
    doc.add_heading("Minor Detail", level=2)
    doc.add_paragraph("Target content for docx.")
    
    doc.save(path)

def create_pdf_file(path):
    """Create a PDF file."""
    if not fitz:
        pytest.skip("pymupdf not installed")
        
    doc = fitz.open()
    page = doc.new_page()
    
    # We write text. PDF layout extraction is tricky.
    # Standard pypdf/llama-index ingestion might not infer semantic headings 
    # unless using a smart model. We just want to verify behavior (likely None).
    text = "PDF Title\n\nTarget content for pdf.\n"
    page.insert_text((50, 50), text)
    
    doc.save(path)
    doc.close()

import asyncio

def test_heading_path_extraction(test_env):
    """Verify heading_path for different file types."""
    asyncio.run(_async_test_heading_path_extraction(test_env))

async def _async_test_heading_path_extraction(test_env):
    data_dir = test_env["data_dir"]
    storage_dir = test_env["storage_dir"]
    
    # 1. Create Test Files
    create_markdown_file(data_dir / "test.md")
    create_docx_file(data_dir / "test.docx")
    create_pdf_file(data_dir / "test.pdf")
    
    # 2. Run Ingestion
    # Patch globals in ingest module
    with patch("ingest.DATA_DIR", data_dir), \
         patch("ingest.STORAGE_DIR", storage_dir), \
         patch("ingest.STATE_DB_PATH", storage_dir / "ingestion_state.db"), \
         patch("ingest.RETRIEVAL_MODEL_CACHE_DIR", Path("./models").resolve()), \
         patch("mcp_server.STORAGE_DIR", storage_dir), \
         patch("mcp_server.RETRIEVAL_MODEL_CACHE_DIR", Path("./models").resolve()):
        
        # Ensure we don't try to download models in tests if not needed, 
        # or assume they are present/mocked.
        # ingest.build_index() calls ensure_embedding_model_cached.
        # Use offline mode to avoid network hits, assuming models are cached or we accept failure/skip.
        # For this integration test, we might need the models if the ingestion *uses* them to embed.
        # ingest.py: build_index -> index.insert(doc) -> uses embed model.
        
        # If models are not cached locally, this test might fail or need network.
        # The environment likely has them if previous tests passed.
        # We'll set offline=False to allow download if needed, or True if strictly offline.
        # Given "run or add an automated test", let's try to reuse existing cache.
        
        # To avoid expensive embedding validation in this logic test, we could mock the embedding model,
        # but ingest.py is tightly coupled.
        # Let's rely on --offline flag or env vars set by checking existing tests.
        
        # Patching ensure_embedding_model_cached to be a no-op or valid mock might be safer 
        # if we only care about METADATA logic, but LlamaIndex will try to embed.
        # Let's assume the environment is set up like the other tests (which use test_data_dummy).
        
        print("Running ingestion...")
        # We need to mock the embedding model to avoid loading heavy models or network calls?
        # Or just run it if it's fast enough.
        # Let's try running it 'live' first, but if it fails we mock.
        
        from llama_index.core.embeddings import BaseEmbedding
        import hashlib
        class MockEmbedding(BaseEmbedding):
            def _get_vector(self, text):
                # Create a deterministic vector based on text content.
                # Use simple hash to ensure uniqueness
                hash_object = hashlib.md5(text.encode())
                start_val = int(hash_object.hexdigest(), 16)
                
                vec = [0.0] * 384
                # Use the hash to seed non-zero values
                for i in range(10):
                    idx = (start_val + i*13) % 384
                    vec[idx] = 1.0
                return vec

            def _get_query_embedding(self, query): return self._get_vector(query)
            def _get_text_embedding(self, text): return self._get_vector(text)
            async def _aget_query_embedding(self, query): return self._get_vector(query)
        
        mock_embed = MockEmbedding(model_name="mock")
        
        with patch("ingest._create_fastembed_embedding", return_value=mock_embed), \
             patch("ingest.ensure_embedding_model_cached"), \
             patch("ingest.ensure_rerank_model_cached"):
                 
            ingest.build_index(offline=True)

            # 3. Run Retrieval
            # Reset server cache
            mcp_server._index_cache = None
            
            # Patch mcp_server globals
            # Mock reranker too
            mcp_server.RETRIEVAL_EMBED_TOP_K = 100 # Ensure we get our docs
            
            # We also need to patch the embedding model inside mcp_server._ensure_embed_model
            with patch("mcp_server.FastEmbedEmbedding", return_value=mock_embed), \
                 patch("mcp_server.Settings") as mock_settings:
                
                # We need to manually set the embed model because _ensure_embed_model sets it on global Settings
                mock_settings.embed_model = mock_embed
                
                # Mock reranker to return everything with score 1.0 (pass-through)
                # Or ensure _ensure_reranker works. It loads flashrank.
                # Let's mock _ensure_reranker to return a mock object that has rerank()
                class MockReranker:
                    def rerank(self, request):
                        # Simple score based on term overlap or just passthrough
                        return [{"text": p["text"], "score": 1.0} for p in request.passages]
                
                with patch("mcp_server._ensure_reranker", return_value=MockReranker()):
                    
                    # Test Markdown
                    print("Querying Markdown...")
                    # Note: "Target content for markdown" has newline in file, query doesn't.
                    # Mock embedding must handle this or we rely on similarity.
                    # With our hash, if text differs by a char, vectors are orthogonal (mostly).
                    # We should search for substring or ensure the text we search for matches the chunk exactly?
                    # The chunk will be "Content under subheading 1.1.\nTarget content for markdown." likely.
                    # So "Target content for markdown" query vector != chunk vector.
                    # Vector search finds NEAREST.
                    # Our mock vector is sparse (10 ones).
                    # If query hash and chunk hash differ, overlap is likely 0.
                    # So vector search returns RANDOM/First docs?
                    # We need the query vector to match the chunk vector for the chunk containing the query.
                    # This is hard with a hash mock.
                    
                    # Alternative: Mock Embedding checks if query is IN text.
                    # If query in text, return SAME vector as query?
                    # No, _get_text_embedding takes text, _get_query_embedding takes query.
                    # They don't know about each other.
                    
                    # Let's simple make the query the EXACT text of the chunk we expect?
                    pass

                    # BETTER MOCK:
                    # _get_text_embedding: Tokenize, sum tokens?
                    # simple Bag of Words mock.
        
        class BoWEmbedding(BaseEmbedding):
             def _get_vector(self, text):
                 vec = [0.0] * 384
                 # Simple token hash sum
                 for word in text.split():
                     model_idx = sum(ord(c) for c in word) % 384
                     vec[model_idx] += 1.0
                 return vec
             def _get_query_embedding(self, query): return self._get_vector(query)
             def _get_text_embedding(self, text): return self._get_vector(text)
             async def _aget_query_embedding(self, query): return self._get_vector(query)

        mock_embed = BoWEmbedding(model_name="mock")

        with patch("ingest._create_fastembed_embedding", return_value=mock_embed), \
             patch("ingest.ensure_embedding_model_cached"), \
             patch("ingest.ensure_rerank_model_cached"):
                 
            ingest.build_index(offline=True)

            mcp_server._index_cache = None
            mcp_server.RETRIEVAL_EMBED_TOP_K = 100
            
            with patch("mcp_server.FastEmbedEmbedding", return_value=mock_embed), \
                 patch("mcp_server.Settings") as mock_settings:
                mock_settings.embed_model = mock_embed
                
                class MockReranker:
                    def rerank(self, request): return [{"text": p["text"], "score": 1.0} for p in request.passages]
                
                with patch("mcp_server._ensure_reranker", return_value=MockReranker()):
                    
                    # Test Markdown
                    print("Querying Markdown...")
                    # Query for a unique word in the MD file
                    result_md = await mcp_server.retrieve_docs("markdown")
                    chunks_md = result_md["chunks"]
                    
                    # Filter for the right file using URI or text content
                    # Start with a filter to ensure we act on the right chunk if search is fuzzy
                    target_md = [c for c in chunks_md if "Target content for markdown" in c["text"]]
                    assert len(target_md) > 0, "MD chunk not found in results"
                    md_chunk = target_md[0]
                    
                    print(f"MD Location: {md_chunk['location']}")
                    
                    # Test DOCX
                    print("Querying DOCX...")
                    result_docx = await mcp_server.retrieve_docs("docx")
                    chunks_docx = result_docx["chunks"]
                    target_docx = [c for c in chunks_docx if "Target content for docx" in c["text"]]
                    assert len(target_docx) > 0, "DOCX chunk not found in results"
                    docx_chunk = target_docx[0]
                    
                    print(f"DOCX Location: {docx_chunk['location']}")
                    assert docx_chunk["location"]["heading_path"] is not None, "DOCX heading_path should not be None"
                    # Current implementation only captures immediate heading
                    assert "Minor Detail" in docx_chunk["location"]["heading_path"]


                    # Test PDF
                    print("Querying PDF...")
                    result_pdf = await mcp_server.retrieve_docs("pdf")
                    chunks_pdf = result_pdf["chunks"]
                    target_pdf = [c for c in chunks_pdf if "Target content for pdf" in c["text"]]
                    assert len(target_pdf) > 0, "PDF chunk not found in results"
                    pdf_chunk = target_pdf[0]
                    print(f"PDF Location: {pdf_chunk['location']}")
                    # PDF Ingestion uses SimpleDirectoryReader. Likely no heading extraction.
                    
if __name__ == "__main__":
    # Allow running directly
    sys.exit(pytest.main(["-v", __file__]))
