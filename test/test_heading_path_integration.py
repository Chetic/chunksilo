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
    """Create a markdown file with nested headings."""
    content = """# Top Level Heading
Introduction paragraph with Lorem ipsum dolor sit amet consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam quis nostrud exercitation ullamco.
Laboris nisi ut aliquip ex ea commodo consequat duis aute irure dolor in reprehenderit in voluptate velit.
Esse cillum dolore eu fugiat nulla pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui.
Officia deserunt mollit anim id est laborum sed ut perspiciatis unde omnis iste natus error sit voluptatem.
Accusantium doloremque laudantium totam rem aperiam eaque ipsa quae ab illo inventore veritatis quasi architecto.
Beatae vitae dicta sunt explicabo nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit.

## Second Level
Brief content here.

### Third Level
Target content for markdown with deep nesting and this is where we test the actual functionality with lots of text.
This is the specific content we're looking for in our test assertions to verify correct behavior and proper heading hierarchy.
It should appear under the full hierarchy of headings showing all three levels properly extracted from the document structure.
Additional sentences here to make this section more substantial and realistic for testing purposes and push boundaries.
More content to ensure this Third Level section is long enough to get its own dedicated chunk during the splitting process.
Even more filler text here to make absolutely sure that chunks starting in this section capture the full heading path correctly.

## Another Second Level
More content in this section with additional paragraphs for completeness.
This provides additional test coverage for the heading extraction logic implementation.
"""
    with open(path, "w") as f:
        f.write(content)

def create_docx_file(path):
    """Create a DOCX file with nested headings."""
    if not Document:
        pytest.skip("python-docx not installed")

    doc = Document()
    doc.add_heading("Chapter 1: Introduction", level=1)
    doc.add_paragraph("Overview of the system with detailed introduction text.")
    doc.add_paragraph("This section provides background and context for understanding the architecture.")

    doc.add_heading("Architecture", level=2)
    doc.add_paragraph("System design details and architectural patterns.")
    doc.add_paragraph("This section describes the overall structure and components of the system.")

    doc.add_heading("Components", level=3)
    doc.add_paragraph("Target content for docx with hierarchy and nested structure.")
    doc.add_paragraph("This specific section contains the test content we're searching for.")

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
    # Use chunk size 200 to ensure proper heading resolution without metadata warnings
    with patch("ingest.DATA_DIR", data_dir), \
         patch("ingest.STORAGE_DIR", storage_dir), \
         patch("ingest.STATE_DB_PATH", storage_dir / "ingestion_state.db"), \
         patch("ingest.RETRIEVAL_MODEL_CACHE_DIR", Path("./models").resolve()), \
         patch("ingest.CHUNK_SIZE", 200), \
         patch("ingest.CHUNK_OVERLAP", 25), \
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

                    # Test Markdown heading path - should have nested hierarchy
                    print("Testing MD heading path...")
                    assert md_chunk["location"]["heading_path"] is not None, "MD should have heading_path"
                    heading_path_md = md_chunk["location"]["heading_path"]
                    assert len(heading_path_md) >= 2, f"MD should have nested path, got {heading_path_md}"
                    assert "Third Level" in heading_path_md, "Should include deepest heading"
                    print(f"✓ MD heading_path has {len(heading_path_md)} levels: {heading_path_md}")

                    # Test DOCX
                    print("Querying DOCX...")
                    result_docx = await mcp_server.retrieve_docs("hierarchy nested structure")
                    chunks_docx = result_docx["chunks"]
                    target_docx = [c for c in chunks_docx if "Target content for docx" in c["text"]]
                    assert len(target_docx) > 0, "DOCX chunk not found in results"
                    docx_chunk = target_docx[0]

                    print(f"DOCX Location: {docx_chunk['location']}")

                    # Test DOCX heading path hierarchy
                    print("Testing DOCX heading path...")
                    assert docx_chunk["location"]["heading_path"] is not None, "DOCX heading_path should not be None"
                    heading_path_docx = docx_chunk["location"]["heading_path"]
                    assert len(heading_path_docx) >= 2, f"DOCX should have nested path, got {heading_path_docx}"
                    assert "Components" in heading_path_docx, "Should include deepest heading"
                    print(f"✓ DOCX heading_path has {len(heading_path_docx)} levels: {heading_path_docx}")


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
