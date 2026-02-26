#!/usr/bin/env python3
"""Tests for chunksilo.server module."""

import asyncio
from unittest.mock import patch

from chunksilo.server import _create_server, _rotate_log_if_needed

# =============================================================================
# Tests for _rotate_log_if_needed
# =============================================================================


class TestRotateLogIfNeeded:
    def test_oversized_log_rotated(self, tmp_path, monkeypatch):
        log_file = tmp_path / "mcp.log"
        # Write >10MB to the log file
        log_file.write_bytes(b"x" * (11 * 1024 * 1024))

        monkeypatch.setattr("chunksilo.server.LOG_FILE", str(log_file))
        _rotate_log_if_needed()

        # Original file should be recreated (empty) and a rotated file should exist
        assert log_file.exists()
        assert log_file.stat().st_size == 0
        rotated = [f for f in tmp_path.iterdir() if f.name.startswith("mcp_")]
        assert len(rotated) == 1

    def test_undersized_log_not_rotated(self, tmp_path, monkeypatch):
        log_file = tmp_path / "mcp.log"
        log_file.write_text("small log")

        monkeypatch.setattr("chunksilo.server.LOG_FILE", str(log_file))
        _rotate_log_if_needed()

        assert log_file.read_text() == "small log"
        rotated = [f for f in tmp_path.iterdir() if f.name.startswith("mcp_")]
        assert len(rotated) == 0

    def test_nonexistent_log_no_error(self, tmp_path, monkeypatch):
        log_file = tmp_path / "mcp.log"
        monkeypatch.setattr("chunksilo.server.LOG_FILE", str(log_file))

        # Should not raise
        _rotate_log_if_needed()
        assert not log_file.exists()


# =============================================================================
# Tests for _create_server
# =============================================================================


class TestCreateServer:
    @patch("chunksilo.server.run_search", create=True)
    def test_returns_fastmcp_with_search_tool(self, mock_search):
        mcp = _create_server()
        # FastMCP instance should have the search_docs tool
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "search_docs" in tool_names

    @patch("chunksilo.search.run_search")
    def test_search_docs_calls_run_search(self, mock_run_search):
        mock_run_search.return_value = {"chunks": [], "matched_files": []}

        mcp = _create_server()

        # Find and invoke the search_docs tool via asyncio
        async def _call():
            result = await mcp._tool_manager.call_tool("search_docs", {"query": "test"})
            return result

        asyncio.run(_call())
        mock_run_search.assert_called_once()
        call_args = mock_run_search.call_args
        assert call_args[0][0] == "test"


# =============================================================================
# Tests for main
# =============================================================================


class TestMain:
    @patch("chunksilo.server.run_server")
    @patch("chunksilo.server._setup_logging")
    def test_main_passes_config(self, mock_logging, mock_run_server):
        from chunksilo.server import main

        with patch("sys.argv", ["chunksilo-mcp", "--config", "/tmp/test.yaml"]):
            main()

        mock_run_server.assert_called_once()
        config_arg = mock_run_server.call_args[1].get("config_path") or mock_run_server.call_args[0][0]
        assert str(config_arg) == "/tmp/test.yaml"

    @patch("chunksilo.server.run_server")
    @patch("chunksilo.server._setup_logging")
    def test_main_no_config(self, mock_logging, mock_run_server):
        from chunksilo.server import main

        with patch("sys.argv", ["chunksilo-mcp"]):
            main()

        mock_run_server.assert_called_once()
        config_arg = mock_run_server.call_args[0][0]
        assert config_arg is None
