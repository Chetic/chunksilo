#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
MCP server for querying documentation using RAG.
Returns raw document chunks for the calling LLM to synthesize.

Two transports:
- stdio (default): the MCP client starts this process itself; one user, one
  machine.
- streamable-http: listens on server.host:server.port for any MCP client that
  can reach it. This server performs NO authentication of its own - keep the
  loopback bind (the default) or put an authenticating reverse proxy in front
  of it before exposing the port. server.public_url tells the transport which
  Host header that proxy forwards; server.authorization_servers lets clients
  discover where to obtain a token.
"""
import argparse
import asyncio
import logging
import logging.handlers
import os
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import __version__

logger = logging.getLogger(__name__)

# Log file configuration. The log lives in the storage directory - never the
# CWD, which on a mis-started service could be inside an indexed tree - and
# is private: it records what was searched.
LOG_FILE_NAME = "mcp.log"
LOG_MAX_SIZE_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5

TRANSPORTS = ("stdio", "streamable-http")

# What the MCP SDK accepts for a loopback bind; kept when a public URL is added.
_LOOPBACK_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
_LOOPBACK_ORIGINS = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]

# Module-level state (set during initialization)
_server_config_path: Path | None = None
_mcp: FastMCP | None = None


class _PrivateRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler whose files are only ever mode 0600."""

    def _open(self):
        fd = os.open(
            self.baseFilename, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
        )
        # O_CREAT does not chmod a file that already exists, so be explicit.
        os.fchmod(fd, 0o600)
        return os.fdopen(fd, self.mode, encoding=self.encoding)


def _setup_logging(storage_dir: str | Path):
    """Configure logging for MCP server mode - file only, no stdout/stderr."""
    log_path = Path(storage_dir).expanduser() / LOG_FILE_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
        handlers=[
            _PrivateRotatingFileHandler(
                log_path,
                maxBytes=LOG_MAX_SIZE_BYTES,
                backupCount=LOG_BACKUP_COUNT,
                encoding="utf-8",
            ),
        ],
    )
    logging.getLogger("llama_index.readers.confluence").setLevel(logging.WARNING)


def _public_url(server_cfg: dict[str, Any]) -> tuple[str, str, str] | None:
    """``(url, scheme, netloc)`` from server.public_url, or None when unset."""
    raw = str(server_cfg.get("public_url") or "").strip().rstrip("/")
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.path:
        raise SystemExit(
            "server.public_url must be an absolute URL with no path, such as "
            f"https://search.example.com - got {raw!r}"
        )
    return raw, parsed.scheme, parsed.netloc


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


def _transport_security(server_cfg: dict[str, Any]) -> TransportSecuritySettings | None:
    """Host/Origin allow-list for the HTTP transport, or None for the SDK default.

    The SDK enables DNS-rebinding protection for a loopback bind and then
    accepts only loopback Host headers (421 otherwise). A reverse proxy forwards
    the public hostname, so the public URL - and anything in allowed_hosts /
    allowed_origins - has to be allowed explicitly; the loopback entries stay so
    the proxy can also address the upstream directly.
    """
    public = _public_url(server_cfg)
    extra_hosts = [str(h) for h in server_cfg.get("allowed_hosts") or []]
    extra_origins = [str(o) for o in server_cfg.get("allowed_origins") or []]
    if public is None and not extra_hosts and not extra_origins:
        return None

    hosts = list(_LOOPBACK_HOSTS)
    origins = list(_LOOPBACK_ORIGINS)
    if public is not None:
        _url, scheme, netloc = public
        parsed = urlparse(f"{scheme}://{netloc}")
        host_only = netloc[: -(len(str(parsed.port)) + 1)] if parsed.port else netloc
        hosts += [netloc, host_only, f"{host_only}:*"]
        origins += [f"{scheme}://{netloc}", f"{scheme}://{host_only}", f"{scheme}://{host_only}:*"]
    bind_host = str(server_cfg.get("host") or "")
    if bind_host and bind_host not in ("127.0.0.1", "localhost", "::1", "0.0.0.0", "::"):
        hosts.append(f"{bind_host}:*")
    hosts += extra_hosts
    origins += extra_origins
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_dedupe(hosts),
        allowed_origins=_dedupe(origins),
    )


def _forwarded_user(ctx: Context | None) -> str | None:
    """Identity a reverse proxy forwarded (X-Forwarded-User / -Email), if any.

    Trusted only because the HTTP bind is the proxy's private upstream, and used
    for the log line alone - never for an authorization decision.
    """
    if ctx is None:
        return None
    try:
        request = ctx.request_context.request
    except (ValueError, AttributeError):
        return None
    if request is None:
        return None
    headers = request.headers
    return headers.get("x-forwarded-user") or headers.get("x-forwarded-email") or None


def _create_server(
    config: dict[str, Any] | None = None, transport: str = "stdio"
) -> FastMCP:
    """Create and configure the MCP server with tools.

    For ``streamable-http`` the bind address, the accepted Host/Origin values
    and the optional protected-resource metadata come from ``config["server"]``;
    the stdio server needs no configuration at all.
    """
    from .search import run_search

    if transport == "streamable-http":
        server_cfg = config["server"]
        mcp = FastMCP(
            "chunksilo",
            host=server_cfg["host"],
            port=int(server_cfg["port"]),
            # Each POST is independent - no session affinity - so a tool call
            # never depends on which worker served the previous one.
            stateless_http=True,
            json_response=True,
            transport_security=_transport_security(server_cfg),
        )
        _add_http_routes(mcp, server_cfg)
    else:
        mcp = FastMCP("chunksilo")

    @mcp.tool()
    async def search_docs(
        query: Annotated[str, Field(description="Search query text")],
        date_from: Annotated[str | None, Field(description="Optional start date filter (YYYY-MM-DD format, inclusive)")] = None,
        date_to: Annotated[str | None, Field(description="Optional end date filter (YYYY-MM-DD format, inclusive)")] = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Search across all your indexed documentation using a natural language query."""
        user = _forwarded_user(ctx)
        if user:
            logger.info("Search by %s", user)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: run_search(query, date_from, date_to, config_path=_server_config_path)
        )

    return mcp


def _add_http_routes(mcp: FastMCP, server_cfg: dict[str, Any]) -> None:
    """Plain HTTP routes next to /mcp: a health check and, when configured,
    the RFC 9728 protected-resource metadata MCP clients use to find the
    authorization server. Neither is authenticated - exempt them at the proxy."""

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_request: Request) -> JSONResponse:
        # The server only starts listening after warm-up, so reachable = ready.
        return JSONResponse({"status": "ok", "version": __version__})

    authorization_servers = [str(u) for u in server_cfg.get("authorization_servers") or []]
    if not authorization_servers:
        return
    public = _public_url(server_cfg)
    if public is None:
        raise SystemExit(
            "server.authorization_servers requires server.public_url: the metadata "
            "must name the URL clients reach this server on"
        )
    mcp_path = mcp.settings.streamable_http_path
    metadata = {
        "resource": public[0] + mcp_path,
        "authorization_servers": authorization_servers,
        "bearer_methods_supported": ["header"],
        "resource_name": "ChunkSilo",
    }

    async def protected_resource(_request: Request) -> JSONResponse:
        return JSONResponse(metadata)

    # Both the origin-wide form and the path-suffixed form of the well-known
    # URL, so a client can look it up either way.
    for path in ("/.well-known/oauth-protected-resource",
                 "/.well-known/oauth-protected-resource" + mcp_path):
        mcp.custom_route(path, methods=["GET"])(protected_resource)


def run_server(config_path: Path | None = None, transport_override: str | None = None):
    """Start the MCP server."""
    global _server_config_path, _mcp

    from .cfgload import load_config

    # Load - and thereby activate - the configuration before importing
    # .search/.index: index.py reads it at import time via load_config().
    config = load_config(config_path)
    if config_path:
        _server_config_path = config_path

    # Logging goes to <storage_dir>/mcp.log, so it could not be configured
    # before the config was read.
    _setup_logging(config["storage"]["storage_dir"])
    logger.info("Starting ChunkSilo MCP server")
    transport = transport_override or config["server"]["transport"]

    if transport == "stdio":
        _mcp = _create_server(config, transport)
        _mcp.run()
        return

    if transport != "streamable-http":
        raise SystemExit(
            f"Unknown server.transport: {transport!r} (expected one of {', '.join(TRANSPORTS)})"
        )

    # Pay for the index, the models and the BM25 index once, before the first
    # request - concurrent first requests would otherwise all wait on the
    # slowest cold start.
    from . import search as search_module

    logger.info("Warming up the search pipeline before accepting requests")
    search_module.warm_up(config)

    _mcp = _create_server(config, transport)
    public = _public_url(config["server"])
    logger.info(
        "Serving MCP over streamable-http on %s:%s%s (no authentication - front it with a proxy before exposing it)",
        config["server"]["host"],
        config["server"]["port"],
        f", public URL {public[0]}" if public else "",
    )
    _mcp.run(transport="streamable-http")


def main():
    """Entry point for the chunksilo-mcp command."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(
        prog="chunksilo-mcp",
        description="Run ChunkSilo MCP server (stdio or streamable-http transport)",
    )
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument(
        "--transport",
        choices=list(TRANSPORTS),
        help="Override server.transport from config",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    run_server(config_path, transport_override=args.transport)


if __name__ == "__main__":
    main()
