#!/usr/bin/env python3
"""Tests for chunksilo.server module."""

import asyncio
from unittest.mock import patch

from chunksilo.server import _create_server, _setup_logging

# =============================================================================
# Tests for _setup_logging
# =============================================================================


class TestSetupLogging:
    def _teardown(self):
        import logging

        # Release the file handler so tmp_path can be removed on Windows and
        # later tests get a clean root logger.
        for handler in logging.getLogger().handlers[:]:
            logging.getLogger().removeHandler(handler)
            handler.close()

    def test_log_lands_in_storage_dir_with_private_mode(self, tmp_path):
        import logging
        import stat

        try:
            _setup_logging(tmp_path / "storage")
            logging.getLogger("chunksilo.test").info("hello")
            log_file = tmp_path / "storage" / "mcp.log"
            assert log_file.exists()
            assert stat.S_IMODE(log_file.stat().st_mode) == 0o600
            assert "hello" in log_file.read_text()
        finally:
            self._teardown()

    def test_existing_wider_mode_is_tightened(self, tmp_path):
        import stat

        storage = tmp_path / "storage"
        storage.mkdir()
        log_file = storage / "mcp.log"
        log_file.write_text("old")
        log_file.chmod(0o644)
        try:
            _setup_logging(storage)
            assert stat.S_IMODE(log_file.stat().st_mode) == 0o600
        finally:
            self._teardown()

    def test_rotation_is_bounded(self, tmp_path):
        import logging

        from chunksilo import server as server_module

        storage = tmp_path / "storage"
        try:
            with patch.object(server_module, "LOG_MAX_SIZE_BYTES", 512), patch.object(
                server_module, "LOG_BACKUP_COUNT", 2
            ):
                _setup_logging(storage)
                log = logging.getLogger("chunksilo.test")
                for _ in range(200):
                    log.info("x" * 64)
            logs = sorted(p.name for p in storage.iterdir())
            assert "mcp.log" in logs
            # backupCount bounds what rotation leaves behind
            assert len(logs) <= 3
        finally:
            self._teardown()


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
