#!/usr/bin/env python3
"""Tests for chunksilo.server module."""

import asyncio
from unittest.mock import patch

import pytest

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

    @patch("chunksilo.server.run_server")
    def test_main_passes_transport_override(self, mock_run_server):
        from chunksilo.server import main

        with patch("sys.argv", ["chunksilo-mcp", "--transport", "streamable-http"]):
            main()

        assert mock_run_server.call_args.kwargs["transport_override"] == "streamable-http"

    def test_main_rejects_unknown_transport(self):
        from chunksilo.server import main

        with patch("sys.argv", ["chunksilo-mcp", "--transport", "carrier-pigeon"]):
            with pytest.raises(SystemExit):
                main()


# =============================================================================
# Tests for the transports
# =============================================================================


class TestTransports:
    HTTP_CONFIG = {
        "server": {"transport": "streamable-http", "host": "127.0.0.1", "port": 8765},
    }

    def test_http_server_binds_the_configured_address(self):
        mcp = _create_server(self.HTTP_CONFIG, "streamable-http")

        assert mcp.settings.host == "127.0.0.1"
        assert mcp.settings.port == 8765
        assert mcp.settings.stateless_http is True
        assert mcp.settings.json_response is True
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "search_docs" in tool_names

    def _run(self, config, monkeypatch, tmp_path, transport_override=None):
        """Run run_server with the network, logging and warm-up stubbed out."""
        from chunksilo import search
        from chunksilo import server as server_module

        config = {**config, "storage": {"storage_dir": str(tmp_path)}}
        calls = {"warm_up": [], "run": []}
        monkeypatch.setattr("chunksilo.cfgload.load_config", lambda *_a, **_k: config)
        monkeypatch.setattr(server_module, "_setup_logging", lambda _d: None)
        monkeypatch.setattr(search, "warm_up", lambda cfg: calls["warm_up"].append(cfg))
        monkeypatch.setattr(
            server_module.FastMCP, "run", lambda self, **kw: calls["run"].append(kw)
        )
        server_module.run_server(None, transport_override=transport_override)
        return calls

    def test_http_transport_warms_up_then_serves(self, monkeypatch, tmp_path):
        calls = self._run(self.HTTP_CONFIG, monkeypatch, tmp_path)

        assert len(calls["warm_up"]) == 1
        assert calls["run"] == [{"transport": "streamable-http"}]

    def test_stdio_transport_stays_lazy(self, monkeypatch, tmp_path):
        config = {"server": {"transport": "stdio", "host": "127.0.0.1", "port": 8400}}

        calls = self._run(config, monkeypatch, tmp_path)

        assert calls["warm_up"] == []
        assert calls["run"] == [{}]

    def test_override_wins_over_config(self, monkeypatch, tmp_path):
        calls = self._run(self.HTTP_CONFIG, monkeypatch, tmp_path, transport_override="stdio")

        assert calls["warm_up"] == []
        assert calls["run"] == [{}]

    def test_unknown_transport_exits(self, monkeypatch, tmp_path):
        config = {"server": {"transport": "carrier-pigeon", "host": "127.0.0.1", "port": 1}}

        with pytest.raises(SystemExit):
            self._run(config, monkeypatch, tmp_path)


# =============================================================================
# Tests for running behind a reverse proxy
# =============================================================================


_INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}
_SEARCH = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {"name": "search_docs", "arguments": {"query": "widget"}},
}


def _http_config(**server):
    base = {"transport": "streamable-http", "host": "127.0.0.1", "port": 8765}
    return {"server": {**base, **server}}


class TestReverseProxySupport:
    def test_without_public_url_the_sdk_loopback_protection_stays(self):
        security = _create_server(_http_config(), "streamable-http").settings.transport_security

        assert security.enable_dns_rebinding_protection is True
        assert "127.0.0.1:*" in security.allowed_hosts
        assert "search.example.com" not in security.allowed_hosts

    def test_public_url_host_and_origin_are_accepted(self):
        config = _http_config(public_url="https://search.example.com")

        security = _create_server(config, "streamable-http").settings.transport_security

        assert {"search.example.com", "search.example.com:*", "127.0.0.1:*"} <= set(security.allowed_hosts)
        assert {"https://search.example.com", "https://search.example.com:*"} <= set(security.allowed_origins)

    def test_public_url_with_a_port(self):
        config = _http_config(public_url="https://search.example.com:8443/")

        security = _create_server(config, "streamable-http").settings.transport_security

        assert {"search.example.com:8443", "search.example.com", "search.example.com:*"} <= set(
            security.allowed_hosts
        )
        assert "https://search.example.com:8443" in security.allowed_origins

    def test_extra_hosts_and_origins_are_added(self):
        config = _http_config(
            host="10.0.0.5",
            allowed_hosts=["search.internal:*"],
            allowed_origins=["https://portal.example.com"],
        )

        security = _create_server(config, "streamable-http").settings.transport_security

        assert {"search.internal:*", "10.0.0.5:*"} <= set(security.allowed_hosts)
        assert "https://portal.example.com" in security.allowed_origins

    def test_public_url_must_be_an_absolute_url_without_path(self):
        for bad in ("search.example.com", "https://search.example.com/mcp"):
            with pytest.raises(SystemExit):
                _create_server(_http_config(public_url=bad), "streamable-http")

    def test_authorization_servers_require_a_public_url(self):
        config = _http_config(authorization_servers=["https://idp.example.com/realms/docs"])

        with pytest.raises(SystemExit):
            _create_server(config, "streamable-http")

    def test_proxied_requests_health_and_metadata(self, caplog):
        """Through a proxy the Host header is the public hostname; the health
        and metadata routes are plain GETs; a forwarded identity is logged."""
        import logging

        from starlette.testclient import TestClient

        config = _http_config(
            public_url="https://search.example.com",
            authorization_servers=["https://idp.example.com/realms/docs"],
        )
        headers = {
            "Host": "search.example.com",
            "Origin": "https://search.example.com",
            "Accept": "application/json, text/event-stream",
        }
        with patch("chunksilo.search.run_search") as run_search:
            run_search.return_value = {"chunks": [], "matched_files": []}
            app = _create_server(config, "streamable-http").streamable_http_app()
            with TestClient(app) as client:
                assert client.post("/mcp", headers=headers, json=_INIT).status_code == 200

                with caplog.at_level(logging.INFO, logger="chunksilo.server"):
                    response = client.post(
                        "/mcp", headers={**headers, "X-Forwarded-User": "alice"}, json=_SEARCH
                    )
                assert response.status_code == 200
                assert run_search.call_args[0][0] == "widget"
                assert "Search by alice" in caplog.text

                assert client.post(
                    "/mcp", headers={**headers, "Host": "evil.example.net"}, json=_INIT
                ).status_code == 421

                health = client.get("/health")
                assert health.status_code == 200 and health.json()["status"] == "ok"

                for path in ("/.well-known/oauth-protected-resource",
                             "/.well-known/oauth-protected-resource/mcp"):
                    metadata = client.get(path).json()
                    assert metadata["resource"] == "https://search.example.com/mcp"
                    assert metadata["authorization_servers"] == ["https://idp.example.com/realms/docs"]

    def test_metadata_route_is_absent_without_authorization_servers(self):
        from starlette.testclient import TestClient

        app = _create_server(_http_config(public_url="https://search.example.com"), "streamable-http").streamable_http_app()
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/.well-known/oauth-protected-resource").status_code == 404

    def test_stdio_search_has_no_forwarded_user(self):
        """The context parameter must not leak into the tool's input schema."""
        mcp = _create_server()
        (tool,) = [t for t in mcp._tool_manager.list_tools() if t.name == "search_docs"]
        assert "ctx" not in tool.parameters["properties"]
