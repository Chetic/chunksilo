# Running behind a reverse proxy

Part of [ChunkSilo](../README.md).

ChunkSilo's HTTP transport authenticates nobody. The intended way to share an
instance is to put an authenticating reverse proxy in front of it and let the
proxy decide who gets through. This page shows that setup with
[oauth2-proxy](https://oauth2-proxy.github.io/oauth2-proxy/) checking users
against a KeyCloak realm, one ChunkSilo instance per body of documents, and
membership of a KeyCloak group deciding who may search that instance.

```text
MCP client ──HTTPS──▶ oauth2-proxy (KeyCloak login, group check) ──▶ chunksilo-mcp
                        https://search.example.com                      127.0.0.1:8400
```

Run one `chunksilo-mcp` and one oauth2-proxy per information domain: each
oauth2-proxy has a single allowed-group setting, so the pair is the unit of
access control. Give each instance its own `config.yaml` (its own
`indexing.directories`, `storage.storage_dir` and `server.port`).

## ChunkSilo side

```yaml
server:
  transport: "streamable-http"
  host: "127.0.0.1"                              # only the proxy can reach it
  port: 8400
  public_url: "https://search.example.com"       # what the proxy serves this instance on
  authorization_servers:                         # where MCP clients get a token
    - "https://keycloak.example.com/realms/docs"
```

- `public_url` is required behind a proxy. The MCP transport rejects requests
  whose `Host` header is not the bind address (`421 Invalid Host header`), and
  a proxy forwards the public hostname. `allowed_hosts` / `allowed_origins`
  add further values if the proxy is reached under more than one name.
- `authorization_servers` makes ChunkSilo serve the standard
  protected-resource metadata at `/.well-known/oauth-protected-resource`
  (and `/.well-known/oauth-protected-resource/mcp`), naming the KeyCloak realm.
  MCP clients that obtain their own token read it to find where to log in.
  ChunkSilo only publishes the document; it never checks a token.
- `GET /health` answers `{"status": "ok"}` once the server is warmed up and
  listening; use it for readiness checks.
- If the proxy forwards `X-Forwarded-User` or `X-Forwarded-Email`, each search
  is logged with that identity in `<storage_dir>/mcp.log`. The headers are
  trusted only because the loopback bind means only the proxy can set them;
  they never affect what a search returns.

Start it with `chunksilo-mcp --config /etc/chunksilo/docs.yaml`, or pass
`--transport streamable-http` to override a config that says `stdio`.

## oauth2-proxy side

The essential settings, as flags (every one has a config-file and
environment-variable form):

```bash
oauth2-proxy \
  --http-address=127.0.0.1:4180 \                 # put TLS termination in front, or use --tls-cert-file
  --upstream=http://127.0.0.1:8400/ \
  --provider=keycloak-oidc \
  --oidc-issuer-url=https://keycloak.example.com/realms/docs \
  --client-id=chunksilo-docs --client-secret=... \
  --redirect-url=https://search.example.com/oauth2/callback \
  --email-domain='*' \
  --allowed-group=/docs-readers \                 # the KeyCloak group that may search this instance
  --api-routes='^/mcp' \                          # JSON-RPC callers get 401, not a login redirect
  --skip-jwt-bearer-tokens=true \                 # accept KeyCloak-issued bearer tokens
  --skip-auth-route='GET=^/health' \
  --skip-auth-route='GET=^/\.well-known/oauth-protected-resource' \
  --pass-user-headers=true \                      # X-Forwarded-User / -Email for the search log
  --cookie-secret=...
```

Two kinds of client then work:

- **Browser-driven sessions** log in through KeyCloak and carry the
  oauth2-proxy cookie.
- **MCP clients that do OAuth themselves** (Claude Code, opencode and other
  clients that support remote servers with authorization) get a token from
  KeyCloak and send it as a bearer token; `--skip-jwt-bearer-tokens`
  validates it against the realm and `--allowed-group` applies to it as well.
  The token must carry the `groups` claim and an audience oauth2-proxy accepts
  (`--oidc-extra-audience` if it is not the client id).

Unauthenticated `/mcp` calls receive a plain 401 from oauth2-proxy. Clients
that follow the MCP authorization flow then read the protected-resource
metadata to find the realm; if a client insists on the `WWW-Authenticate`
header naming that metadata, add it on the TLS terminator (nginx:
`add_header WWW-Authenticate 'Bearer resource_metadata="https://search.example.com/.well-known/oauth-protected-resource"' always;`
on the 401 response).

## KeyCloak side

- A confidential client for oauth2-proxy (`chunksilo-docs` above) with the
  callback URL as a valid redirect URI.
- A **Group Membership** mapper that puts the user's groups into the tokens as
  a `groups` claim; oauth2-proxy compares `--allowed-group` against it. Group
  names in KeyCloak are paths (`/docs-readers`).
- For MCP clients that log in by themselves: either allow dynamic client
  registration in the realm (client registration policies) or create a public
  client with PKCE and the redirect URIs those clients use, and map the
  audience so the token is accepted by oauth2-proxy.

## Client side

Point the client at the proxy's URL, not at ChunkSilo directly:

```bash
claude mcp add --transport http chunksilo https://search.example.com/mcp
```

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "chunksilo": {
      "type": "remote",
      "url": "https://search.example.com/mcp",
      "enabled": true
    }
  }
}
```

## Checking the deployment

1. `curl https://search.example.com/health` returns `{"status":"ok"}`.
2. `curl https://search.example.com/.well-known/oauth-protected-resource`
   returns the realm URL.
3. An unauthenticated `POST https://search.example.com/mcp` returns 401.
4. `<storage_dir>/mcp.log` on the ChunkSilo host shows `Serving MCP over
   streamable-http ... public URL https://search.example.com` at startup and
   `Search by <user>` lines afterwards.
5. A `421 Invalid Host header` in the client, or `Invalid Host header:` in
   `mcp.log`, means `server.public_url` does not match the hostname the proxy
   forwards.
