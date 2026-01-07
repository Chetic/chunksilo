
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os
from pathlib import Path

# Add project root to sys.path to import mcp_server
sys.path.append(str(Path(__file__).parent.parent))

from mcp_server import _process_and_format_results, search_local, search_confluence

class TestMCPServer(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Common setup if needed
        pass

    @patch('mcp_server._ensure_reranker')
    def test_process_and_format_results_formatting(self, mock_ensure_reranker):
        # Mock reranker to just return results as is - or simpler, avoiding reranker for this test
        # If we pass nodes, it enters reranking block.
        # Let's mock the reranker to return the same passages with scores.
        
        mock_ranker = MagicMock()
        mock_ensure_reranker.return_value = mock_ranker
        mock_ranker.rerank.side_effect = lambda req: [{"text": p["text"], "score": 0.9} for p in req.passages]
        
        # Create mock nodes
        mock_node = MagicMock()
        mock_node.node.get_content.return_value = "Test content"
        mock_node.node.metadata = {"file_name": "test.txt", "page_label": "1"}
        mock_node.score = 0.5
        
        nodes = [mock_node]
        query = "test"
        start_time = 0.0
        
        result = _process_and_format_results(nodes, query, start_time)
        
        self.assertEqual(result["query"], "test")
        self.assertEqual(len(result["chunks"]), 1)
        self.assertEqual(result["chunks"][0]["text"], "Test content")
        self.assertEqual(result["chunks"][0]["metadata"]["file_name"], "test.txt")
        # Check source list
        self.assertEqual(len(result["sources"]), 1)
        self.assertIn("test.txt", result["sources"][0]["name"])

    @patch('mcp_server._preprocess_query')
    @patch('mcp_server.load_llamaindex_index')
    @patch('mcp_server._process_and_format_results')
    async def test_search_local(self, mock_process, mock_load_index, mock_preprocess):
        # Setup mocks
        mock_preprocess.return_value = "processed query"
        
        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever
        mock_load_index.return_value = mock_index
        
        mock_nodes = [MagicMock()]
        mock_retriever.retrieve.return_value = mock_nodes
        
        mock_process.return_value = {"chunks": ["mocked result"]}
        
        # Run function
        result = await search_local("query")
        
        # Verify calls
        mock_preprocess.assert_called_with("query")
        mock_load_index.assert_called_once()
        mock_retriever.retrieve.assert_called_with("processed query")
        mock_process.assert_called_with(mock_nodes, "processed query", unittest.mock.ANY)
        self.assertEqual(result, {"chunks": ["mocked result"]})

    @patch('os.getenv')
    @patch('mcp_server._preprocess_query')
    @patch('mcp_server._search_confluence')
    @patch('mcp_server._process_and_format_results')
    async def test_search_confluence(self, mock_process, mock_search_conf, mock_preprocess, mock_getenv):
        # Setup mocks
        mock_getenv.side_effect = lambda k, d=None: "https://confluence.example.com" if k == "CONFLUENCE_URL" else d
        mock_preprocess.return_value = "processed query"
        
        mock_nodes = [MagicMock()]
        # _search_confluence is called in run_in_executor, so we need to mock that behavior or just the function result
        # Since we patch the function used inside run_in_executor, but we can't easily patch run_in_executor directly with AsyncMock to wait for it properly if it wasn't mocked at loop level.
        # Actually simplest is to patch _search_confluence and rely on it being called.
        mock_search_conf.return_value = mock_nodes
        
        mock_process.return_value = {"chunks": ["mocked result"]}

        # Run function
        result = await search_confluence("query")
        
        # Verify calls
        mock_preprocess.assert_called_with("query")
        # _search_confluence should be called. 
        # Note: run_in_executor calls the function.
        mock_search_conf.assert_called_with("processed query")
        mock_process.assert_called_with(mock_nodes, "processed query", unittest.mock.ANY)
        self.assertEqual(result, {"chunks": ["mocked result"]})

    @patch('os.getenv')
    @patch('mcp_server._process_and_format_results')
    async def test_search_confluence_disabled(self, mock_process, mock_getenv):
        # Test when CONFLUENCE_URL is not set
        mock_getenv.return_value = None
        
        await search_confluence("query")
        
        # Verify _process_and_format_results called with empty list
        mock_process.assert_called_with([], unittest.mock.ANY, unittest.mock.ANY)

if __name__ == '__main__':
    unittest.main()
