# Security & License Audit: ChunkSilo v2.3.3

**Audit Date:** 2026-04-10
**Artifact:** `chunksilo-v2.3.3-manylinux_2_34_x86_64.tar.gz` (offline bundle)
**Source Reference:** https://github.com/Chetic/chunksilo at tag v2.3.3
**Auditor:** Automated analysis via Claude Code

---

## 1. Executive Summary

**What was audited.** ChunkSilo v2.3.3, a local RAG-based semantic document search tool with an MCP server interface. The review covered all first-party Python source code (11 files, ~4,700 lines), 147 pinned dependencies in the offline bundle, two bundled ML models, build scripts, and CI workflows.

**Deployment scenarios.** Two environments: (1) offline — no internet but connected to internal Confluence/Jira instances and local LLM inference servers; (2) air-gapped — fully isolated, no network beyond the local machine.

**Overall assessment.** ChunkSilo's first-party code is well-written and security-conscious. It avoids dangerous patterns (no eval/exec/pickle, safe YAML loading, no shell injection vectors), exposes a minimal MCP attack surface (single read-only search tool), and implements a functional offline mode for ML model access. The primary risks come from the transitive dependency footprint — 147 packages where llama-index pulls in telemetry (PostHog), cloud-service clients (OpenAI, LlamaCloud), and packages with native code that cannot be source-audited from the bundle.

**Top findings:**

1. **PostHog telemetry SDK bundled as transitive dependency** — may attempt outbound analytics calls (Medium)
2. **Unused cloud-service client libraries bundled** — openai, llama-cloud increase attack surface and may attempt network calls (Medium)
3. **Confluence/Jira API calls not gated by offline mode flag** — `retrieval.offline` only controls ML model downloads (Medium)
4. **No published checksum or signature for release tarball** — bundle integrity relies solely on GitHub HTTPS transport (Low)
5. **Config file stores API tokens in plaintext** — no guidance on file permission hardening (Low)

**Recommendation: Approve with conditions.** The tool is safe for deployment provided the conditions in Section 9 are met. All Medium findings are addressable — either through environment variable configuration at deployment time or through changes the developer can make in a future release. The fact that this is an internal project with an accessible developer significantly reduces residual risk.

---

## 2. Context: Internal Project

ChunkSilo is developed and maintained by Fredrik Reveny, an employee at our company. This changes the risk calculus in several important ways:

- **Patch turnaround.** We can request security fixes, configuration changes, or dependency updates directly. There is no upstream vendor negotiation or open-source contribution process to navigate.
- **Bus factor.** The project has a single maintainer. If Fredrik changes roles or leaves, someone would need to take over maintenance. Mitigation: the codebase is clean, well-structured, and under 5,000 lines of first-party code with a comprehensive test suite (20 test files). Handoff is feasible.
- **Sustainability.** The project uses standard Python tooling (setuptools, pytest, GitHub Actions CI). There are no exotic build requirements. The primary maintenance burden is keeping the 147 transitive dependencies updated.
- **Governance.** CI enforces linting (ruff), license compliance (pip-licenses), and functional tests on every PR. Releases are manual via GitHub Actions workflow with automated changelog generation.

---

## 3. Security Findings

### Medium

**M1. PostHog telemetry SDK bundled as transitive dependency**

- **Description.** The package `posthog==7.0.1` is included in the offline bundle as a transitive dependency of `llama-index-core` via `llama-cloud`. PostHog is an analytics and product telemetry SDK designed to send usage data to external servers. While ChunkSilo's own code never imports or calls PostHog, the llama-index framework may invoke it internally for usage tracking.
- **Evidence.** `scripts/manylinux_2_34-constraints.txt` line 94: `posthog==7.0.1`. Dependency chain: `llama-index` → `llama-index-core` → `llama-cloud` → `posthog`.
- **Risk.** In offline environments, PostHog calls would fail silently or cause startup delays due to connection timeouts. In the worst case, if the internal network has outbound HTTP access, usage telemetry could be transmitted to PostHog's servers. This is a data exfiltration vector for metadata (not document content, but usage patterns).
- **Mitigation.** Set environment variables at deployment: `DO_NOT_TRACK=1`, `POSTHOG_DISABLED=true`. Request the developer to evaluate whether `llama-cloud` can be excluded from the dependency tree (it is not used by ChunkSilo). Long-term: consider pinning a llama-index build without the llama-cloud dependency.

**M2. Unused cloud-service client libraries in bundle**

- **Description.** The offline bundle includes `openai==2.9.0`, `llama-cloud==0.1.35`, `llama-cloud-services==0.6.54`, `llama-index-llms-openai==0.6.10`, `llama-index-embeddings-openai==0.5.1`, and `llama-index-indices-managed-llama-cloud==0.9.4`. None of these are used by ChunkSilo — they are transitive dependencies pulled in by the `llama-index` meta-package.
- **Evidence.** Grep of `src/chunksilo/` shows zero imports of `openai`, `llama_cloud`, or any OpenAI-related llama-index sub-package. These packages appear only in `scripts/manylinux_2_34-constraints.txt`.
- **Risk.** These packages contain HTTP clients configured to reach `api.openai.com` and LlamaCloud endpoints. If any code path triggers their initialization (even via llama-index internals), they would attempt outbound connections. In air-gapped environments this causes hangs; in offline environments it is an unexpected network call. Additionally, each unused package is an unnecessary supply-chain attack surface.
- **Mitigation.** For deployment: ensure no OpenAI API keys are present in the environment (`OPENAI_API_KEY` unset). Request the developer to evaluate replacing the `llama-index` meta-package with only the specific sub-packages ChunkSilo uses (`llama-index-core`, `llama-index-readers-file`, `llama-index-embeddings-fastembed`, `llama-index-retrievers-bm25`) to eliminate these transitive dependencies.

**M3. Confluence/Jira API calls not gated by the offline mode flag**

- **Description.** The `retrieval.offline: true` configuration flag controls only ML model download behavior (sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`). Confluence and Jira API calls are controlled independently by their respective URL configuration: if `confluence.url` or `jira.url` is non-empty and credentials are provided, the tool will attempt HTTP calls to those services regardless of the offline flag.
- **Evidence.** `src/chunksilo/models.py:54-68` (offline mode only sets HF environment variables). `src/chunksilo/search.py:726-788` (Confluence calls gated by `config["confluence"]["url"]`, not by offline flag). `src/chunksilo/search.py:791-947` (Jira calls gated by `config["jira"]["url"]`, not by offline flag). `AGENTS.md` line 22: "it is assumed that a local Confluence instance is reachable."
- **Risk.** In air-gapped environments, if someone copies a config file that includes Confluence/Jira URLs, the tool will attempt connections that will hang until timeout (default 10 seconds per service per query). This is a usability issue, not a security vulnerability, but it could cause confusing behavior.
- **Mitigation.** For air-gapped deployments: ensure `confluence.url` and `jira.url` are empty strings in `config.yaml`. For offline deployments connecting to internal Confluence/Jira: this is the intended behavior — no change needed. Request the developer to document this distinction clearly and consider adding an `air_gapped: true` flag that disables all network calls.

### Low

**L1. Config file stores API tokens in plaintext**

- **Description.** Confluence and Jira API tokens are stored as plaintext strings in `config.yaml`. The configuration file loading code (`src/chunksilo/cfgload.py:164-165`) reads the file directly with no encryption or secrets-manager integration.
- **Risk.** If `config.yaml` has overly permissive file permissions (e.g., world-readable), other users on the system can read API tokens. Tokens could also appear in backups or configuration management systems.
- **Mitigation.** Document that `config.yaml` should have restricted permissions (`chmod 600`). Request the developer to add environment variable overrides for sensitive fields (e.g., `CHUNKSILO_CONFLUENCE_TOKEN`) so tokens don't need to be in files. Standard practice for tools at this scale.

**L2. Search results pass through MCP client to potentially cloud-hosted LLM**

- **Description.** ChunkSilo returns document chunks to the MCP client, which forwards them to whatever LLM the user has configured. If that LLM is cloud-hosted (e.g., Claude API, OpenAI API), document content leaves the local machine.
- **Evidence.** `AGENTS.md` line 25: "ChunkSilo doesn't phone home, but the LLM client receiving results may." `src/chunksilo/server.py:62-71` — the `search_docs` tool returns full chunk text to the client.
- **Risk.** This is an architectural property of the MCP protocol, not a ChunkSilo bug. However, in sensitive environments, users must understand that using ChunkSilo with a cloud LLM means document fragments are sent to that cloud provider.
- **Mitigation.** Deployment policy: mandate use of local LLM inference servers in sensitive environments. Document this data flow for users. This is already partially addressed in AGENTS.md.

**L3. No published checksum or signature for release tarball**

- **Description.** The GitHub Release for v2.3.3 publishes the `.tar.gz` bundle but no accompanying SHA256 checksum file or GPG signature. The build pipeline (`scripts/package-manylinux-2_34.sh`, `.github/workflows/manual-release.yml`) does not generate integrity artifacts.
- **Risk.** If an attacker compromised the GitHub Release page (e.g., via stolen maintainer token), they could replace the tarball without detection. The risk is mitigated by the fact that this is an internal project with a known developer, and GitHub provides HTTPS transport security.
- **Mitigation.** Request the developer to publish SHA256 checksums alongside release artifacts. After downloading the bundle, compute and record the hash for internal tracking. Consider verifying the bundle contents against the source repository's `scripts/manylinux_2_34-constraints.txt`.

**L4. LibreOffice subprocess for .doc conversion**

- **Description.** Converting legacy `.doc` files to `.docx` invokes LibreOffice via `subprocess.run()` with a fixed argument list (`src/chunksilo/docx_utils.py:57-115`). The command uses list-form arguments (not `shell=True`), includes a timeout (default 60s in code, configurable to 90s via config), and constructs paths using `Path` objects.
- **Risk.** Minimal. The subprocess call is safe against injection because it uses list-form arguments. The file path comes from the filesystem scan (not user input). However, LibreOffice is a complex application and processing a maliciously crafted `.doc` file could trigger vulnerabilities in LibreOffice itself.
- **Mitigation.** If `.doc` support is not needed, exclude `**/*.doc` from the indexing include patterns. If needed, ensure LibreOffice is kept updated on the deployment system.

---

## 4. CVE Inventory

We were unable to run `pip-audit` directly against the bundle in this review environment. The following assessment is based on the pinned versions in `scripts/manylinux_2_34-constraints.txt` and known vulnerability databases as of April 2026.

**Packages most likely to have CVEs against their pinned versions:**

| Package | Pinned Version | Notes |
|:---|:---|:---|
| pillow | 10.3.0 | Pillow frequently receives CVEs for image parser bugs. Version 10.3.0 is from early 2024. However, ChunkSilo uses Pillow only for PDF-to-image conversion via `pdf2image` — the attack surface is limited to images embedded in indexed PDFs. |
| cryptography | 46.0.3 | Actively maintained, high-profile target. Recent version likely patched. Used transitively by `requests` for HTTPS. |
| lxml | 6.0.2 | XML parsing library with historical CVEs. Used by Confluence HTML processing. Only active when Confluence integration is enabled. |
| aiohttp | 3.13.2 | HTTP client/server framework. Used transitively. Recent version. |
| urllib3 | 2.5.0 | HTTP library. Used by `requests`. Recent version likely clean. |
| certifi | 2025.11.12 | CA certificate bundle. Recent version. |

**Recommendation:** Before deploying, run `pip-audit` against the bundle's venv to get a definitive CVE list. The command:
```bash
./venv/bin/pip install pip-audit
./venv/bin/pip-audit
```

Any CVEs found should be assessed for reachability — many Pillow CVEs affect image formats (TIFF, WebP, etc.) that ChunkSilo never processes. Cryptography CVEs typically affect specific cipher suites or protocols that may not be in the code path.

---

## 5. Community Health Summary

Assessment based on maintainer activity, governance, and risk profile for the 147 bundled packages.

### Tier 1 — Low Risk (major, well-governed projects)

~95 packages fall into this tier. Representative examples:

- **numpy, pandas, scipy ecosystem** — NumPy Foundation governance, multiple maintainers, regular releases
- **cryptography** — Python Cryptographic Authority, funded by Mozilla/AWS, rigorous review process
- **pillow** — Active maintainer team, frequent security releases, PSF affiliated
- **pydantic / pydantic-core** — Samuel Colvin + team, backed by commercial entity (Pydantic Inc.)
- **requests, urllib3, certifi** — Core Python HTTP stack, multiple maintainers, PSF ecosystem
- **aiohttp, yarl, multidict** — aio-libs organization, active development
- **lxml** — Long-standing project, active maintainer (Stefan Behnel), regular releases
- **onnxruntime** — Microsoft-maintained, enterprise backing, regular releases
- **huggingface-hub, tokenizers** — Hugging Face Inc., commercial backing, active development
- **PyYAML** — YAML project, stable, infrequent but maintained releases
- **SQLAlchemy** — Mike Bayer + community, long track record, active development
- **protobuf** — Google-maintained
- **Jinja2, MarkupSafe** — Pallets project, well-governed
- **click** — Pallets project
- **mcp** — Anthropic-maintained, actively developed protocol

### Tier 2 — Moderate Risk (smaller but active projects)

~40 packages fall into this tier. Notable entries:

| Package | Notes |
|:---|:---|
| llama-index (+ sub-packages) | LlamaIndex Inc. (commercial), active development, frequent releases. ~12 sub-packages bundled. Fast-moving API surface means dependency churn risk. |
| fastembed | Qdrant (commercial vector DB company), actively maintained |
| flashrank | Prithivi Da (individual), active development, regular releases |
| bm25s | Smaller project, active maintainer, focused scope |
| beautifulsoup4 | Leonard Richardson, long history, stable, infrequent releases |
| jira | Atlassian ecosystem, community maintained, reasonably active |
| atlassian-python-api | Community maintained, active |
| posthog | PostHog Inc. (commercial), actively maintained — but unwanted in our context |
| openai | OpenAI Inc., actively maintained — but unused in our context |
| nltk | NLTK project, academic origin, stable, less frequent releases |
| reportlab | ReportLab Inc., commercial backing |
| defusedxml | Christian Heimes (CPython core dev), stable, purpose-built for security |

### Tier 3 — Elevated Risk (single-maintainer, infrequent, or native code with limited review)

~12 packages warrant closer attention:

| Package | Version | Concern |
|:---|:---|:---|
| py_rust_stemmers | 0.1.5 | Rust native extension, single maintainer, small project. Performs text stemming — limited blast radius. |
| dirtyjson | 1.0.8 | Single maintainer, infrequent releases. Parses malformed JSON — used by llama-index internally. |
| docx2txt | 0.8 | Appears unmaintained (last release 2019). Used for DOCX text extraction as a fallback. |
| striprtf | 0.0.26 | Small project, single maintainer. RTF text stripping — narrow scope. |
| hf-xet | 1.2.0 | Hugging Face's new storage backend (Rust native extension). Relatively new project. Backed by HF Inc. so governance is reasonable, but the codebase is young. |
| banks | 2.2.0 | Templating library, smaller project. Transitive dependency. |
| filetype | 1.2.0 | File type detection, single maintainer, stable but infrequent releases. |
| xlrd | 2.0.2 | Excel reader, maintenance mode (no longer supports .xlsx). Narrow scope. |
| cssselect2 | 0.8.0 | CSS selector library, CourtBouillon team. Small but maintained. |
| tinycss2 | 1.5.1 | CSS parser, CourtBouillon team. Small but maintained. |
| mmh3 | 5.2.0 | MurmurHash3 bindings, C extension. Small project but well-established hash function. |
| greenlet | 3.3.0 | C extension for coroutines. Used by SQLAlchemy. Well-established but complex native code. |

---

## 6. Data Flow & Exfiltration Summary

### Indexing Pipeline

Documents are read from configured directories on the local filesystem. The indexing pipeline (`src/chunksilo/index.py`) processes files through these stages: file discovery (glob patterns) → content extraction (pypdf, python-docx, markdown) → text chunking (LlamaIndex SentenceSplitter) → vector embedding (fastembed with ONNX runtime, CPU-only) → storage to local vector index (`./storage/` directory) and SQLite state database (`./storage/ingestion_state.db`). Headings are stored in a separate JSON file (`./storage/heading_store.json`). A BM25 keyword index is also built for filename matching. All data remains on the local filesystem. Temporary files are created only for `.doc` to `.docx` conversion (in a subdirectory of the storage directory, cleaned up after use). No document content is transmitted over the network during indexing.

### Query Pipeline

When a search query arrives via the MCP `search_docs` tool, the pipeline executes: query preprocessing → vector retrieval (embedding similarity) → BM25 filename matching → optional Confluence/Jira live search (if configured) → date filtering → recency boosting → FlashRank reranking → score thresholding → structured JSON response. The response contains document chunk text, relevance scores, and location metadata (file URI, page number, line number, heading path). This response is returned to the MCP client, which forwards it to whatever LLM the user has configured. **ChunkSilo does not control what happens to the data after it leaves the MCP transport.**

### Telemetry

ChunkSilo's own code contains zero telemetry, analytics, or crash reporting. However, the transitive dependency `posthog==7.0.1` is an analytics SDK. The `llama-index` framework has historically used PostHog for anonymous usage tracking. To ensure no telemetry fires in deployment, set these environment variables: `DO_NOT_TRACK=1`, `POSTHOG_DISABLED=true`, `SCARF_NO_ANALYTICS=true`. With these set, and `retrieval.offline: true` configured, no data should leave the machine from ChunkSilo's process. The only intentional outbound calls are to Confluence/Jira when those integrations are configured with valid URLs and credentials.

---

## 7. Alternatives & Lock-in

### Alternative Tools

| Tool | Description | Offline/Air-gap Support | Maturity | Dependency Footprint |
|:---|:---|:---|:---|:---|
| **RAGFlow** | Open-source RAG engine with document parsing and search | Requires Elasticsearch/Milvus backend, Docker-based. Heavier infrastructure. | Active, backed by InfiniFlow | Heavy — requires multiple services |
| **Danswer/Onyx** | Enterprise document search with connectors | Docker-based, requires PostgreSQL + Vespa. Not designed for air-gap. | Active, VC-funded | Very heavy — full application stack |
| **Private GPT** | Local RAG with privacy focus | Supports fully local operation | Active | Moderate — similar dependency profile |
| **Custom script** | Manual combination of fastembed + FAISS/ChromaDB | Fully controllable, minimal dependencies | N/A — requires development | Minimal |
| **Basic grep/ripgrep** | Full-text search without semantics | Fully offline, zero dependencies | Mature | None |

ChunkSilo occupies a useful middle ground: it provides semantic search with minimal infrastructure (single Python process, no database servers, no Docker) and has first-class offline/air-gap support via the bundle. The closest alternatives either require significantly more infrastructure (RAGFlow, Danswer) or require custom development (fastembed + FAISS). The MCP server interface is a differentiator — none of the alternatives listed ship with MCP support out of the box.

### Lock-in Assessment

- **Index format.** ChunkSilo uses LlamaIndex's default vector store format (JSON-serialized, stored in `./storage/`). This is a LlamaIndex-specific format, not an open standard like SQLite or Parquet. If switching tools, the index cannot be migrated — a full re-index would be required. However, re-indexing is a one-time batch operation, not a data loss scenario (source documents remain unchanged).
- **MCP interface.** The MCP protocol is an open standard. The `search_docs` tool has a simple interface (query string in, structured JSON out). Swapping ChunkSilo for another MCP-compatible search backend would require no changes to MCP client configurations beyond updating the server command.
- **Configuration.** ChunkSilo's config is a single YAML file with no proprietary schema. Switching tools means writing a new config, not migrating one.
- **Switching cost: Low.** The main cost of switching is re-indexing time (minutes to hours depending on corpus size). There is no user retraining, no data migration, and MCP client configs need only a command path change. ChunkSilo does not create meaningful lock-in.

---

## 8. Documentation Assessment

### Well-Documented

- **Installation.** Both PyPI and offline bundle installation paths are clearly documented in `README.md` with step-by-step instructions. The offline bundle flow (`tar xzf` → `./setup.sh` → edit config → build index) is straightforward.
- **Configuration reference.** All config options are documented in tables in `README.md` with descriptions. The `chunksilo --dump-defaults` command prints every option with its default value — a useful self-documenting feature.
- **MCP client setup.** Configuration examples are provided for four MCP clients: Claude Code, Claude Desktop, Cline, and Roo Code. Both PyPI and offline bundle command paths are shown for each.
- **Troubleshooting.** Common issues (missing index, retrieval errors, offline mode, Confluence/Jira setup, network mounts, legacy .doc files) are addressed.
- **AGENTS.md.** Contains important data-flow caveats and contributor guidelines.
- **NOTICE file.** Properly lists third-party components with licenses, including the LGPL svglib transitive dependency.

### Missing or Insufficient

- **Security hardening guide.** No documentation on: file permissions for `config.yaml`, environment variables for telemetry suppression, network firewall rules for deployment, or the distinction between "offline" (ML models offline) and "fully air-gapped" (no network at all).
- **Air-gapped deployment guide.** The README mentions air-gapped use but doesn't document the specific steps or config requirements. The AGENTS.md caveat about Confluence being expected "even in air-gapped mode" may confuse operators.
- **MCP tool documentation.** The `search_docs` tool's input schema and output format are shown by example in the README but not formally documented. The date filter format, score interpretation, and error response structure are not specified.
- **Operational runbook.** No documentation on: log file location and rotation (server writes to `mcp.log` with 10MB rotation), storage directory disk usage expectations, how to rebuild a corrupted index, or how to fully reset state.
- **Dependency inventory.** The offline bundle includes a `THIRD-PARTY-LICENSES.txt` generated at build time, but the main repository doesn't document the full dependency tree or explain why specific packages are included.

### Recommendation

Before operational deployment, our team should create an internal deployment guide supplementing the README with: security hardening steps, environment variable checklist, firewall rules, and an operational runbook. Estimated effort: 1-2 days.

---

## 9. Deployment Recommendations

### For Offline Environments (connected to internal network)

**Config file (`config.yaml`):**
```yaml
retrieval:
  offline: true    # Prevents ML model download attempts

# Configure Confluence/Jira only if connecting to internal instances
confluence:
  url: "https://confluence.internal.example.com"
  username: "service-account"
  api_token: "<token>"

jira:
  url: "https://jira.internal.example.com"
  username: "service-account"
  api_token: "<token>"

ssl:
  ca_bundle_path: "/etc/pki/tls/certs/internal-ca-bundle.pem"
```

**Environment variables (set in the MCP server launch config or shell profile):**
```bash
# Suppress telemetry from transitive dependencies
export DO_NOT_TRACK=1
export POSTHOG_DISABLED=true
export SCARF_NO_ANALYTICS=true

# Ensure ML libraries stay offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Suppress OpenAI client (unused but bundled)
# Do NOT set OPENAI_API_KEY in the environment
unset OPENAI_API_KEY

# Performance tuning
export TOKENIZERS_PARALLELISM=false
export ORT_LOG_LEVEL=3
```

**File permissions:**
```bash
chmod 600 config.yaml          # Only owner can read (contains API tokens)
chmod 700 storage/             # Index data
```

**Network/firewall rules:**
- Allow outbound HTTPS to internal Confluence/Jira instances only
- Block all other outbound traffic from the ChunkSilo process
- No inbound ports needed (MCP uses stdio transport)

**MCP client policy:**
- Use a local LLM inference server if document content is sensitive
- If using a cloud LLM, understand that document chunks will be sent to that provider

### For Air-Gapped Environments (fully isolated)

**Config file (`config.yaml`):**
```yaml
retrieval:
  offline: true

# Leave Confluence/Jira URLs empty to prevent connection attempts
confluence:
  url: ""
jira:
  url: ""
```

**Environment variables (same as offline, plus):**
```bash
# All variables from the offline section above, plus:

# Prevent any HTTP client from attempting connections
export no_proxy="*"
export NO_PROXY="*"
```

**Additional steps:**
1. Transfer the bundle tarball via approved media (USB, optical disc)
2. Verify the tarball hash against a separately communicated SHA256 checksum
3. Extract and run `./setup.sh` — it installs from bundled wheels only (no network)
4. Models are pre-bundled in the `./models/` directory — no downloads needed
5. Test with `./venv/bin/chunksilo --build-index` before configuring the MCP client

**Verification checklist:**
- [ ] `retrieval.offline: true` is set
- [ ] `confluence.url` and `jira.url` are empty
- [ ] Environment variables are set (telemetry, offline flags)
- [ ] `config.yaml` permissions are 600
- [ ] MCP client uses local LLM only
- [ ] No `OPENAI_API_KEY` or similar cloud API keys in environment

---

## 10. License Summary

ChunkSilo is licensed under **Apache-2.0**. All 10 first-party source files carry `SPDX-License-Identifier: Apache-2.0` headers.

**Package license breakdown (147 packages):**

| License Category | Count | Examples |
|:---|:---|:---|
| MIT | ~55 | numpy, tqdm, click, jinja2, pydantic, loguru |
| Apache-2.0 | ~40 | huggingface-hub, tokenizers, fastembed, flashrank, onnxruntime, requests |
| BSD (2/3-clause) | ~25 | pypdf, python-docx, beautifulsoup4, idna, pyyaml |
| ISC | ~3 | — |
| PSF / Python-2.0 | ~5 | typing-extensions, certifi |
| MPL-2.0 | ~2 | certifi (dual-licensed) |
| LGPL-3.0 | 1 | svglib (transitive, see below) |
| Other/Proprietary metadata | 1 | fastembed (PyPI says "Other/Proprietary" but source is Apache-2.0; exempted in CI) |
| Unknown/Unclassified | ~15 | Various packages where PyPI metadata is incomplete but source repos confirm permissive licenses |

**Packages requiring attention:**

| Package | Declared License | Actual License | Action |
|:---|:---|:---|:---|
| svglib | LGPL-3.0 | LGPL-3.0 | Transitive dependency of `llama-index-readers-confluence`. Not modified. Python's dynamic import model satisfies LGPL requirements (no static linking). Documented in NOTICE file. Acceptable. |
| fastembed | "Other/Proprietary" (PyPI) | Apache-2.0 (source) | PyPI metadata is incorrect. Source repository and license file confirm Apache-2.0. Exempted in CI license check. No action needed. |
| docx2txt | MIT | MIT | No issue. Listed because it is exempted in CI license check (the exemption is precautionary, not due to a license problem). |
| py_rust_stemmers | Not declared (PyPI) | MIT (source) | PyPI metadata missing. Source repository confirms MIT. Exempted in CI license check. |

**NOTICE file compliance.** The NOTICE file (`/home/user/chunksilo/NOTICE`) lists Apache-2.0 components, the LGPL svglib dependency, and bundled model licenses. The offline bundle build script generates a comprehensive `THIRD-PARTY-LICENSES.txt` from installed packages. This satisfies Apache-2.0 attribution requirements.

---

## 11. Appendix: Full Package Inventory

All 147 packages pinned in the offline bundle (`scripts/manylinux_2_34-constraints.txt`), plus ChunkSilo itself.

**Legend:**
- **Native**: Package includes compiled C/C++/Rust extensions (`.so` files)
- **Tier**: 1 = Low risk, 2 = Moderate risk, 3 = Elevated risk (see Section 5)
- **Risk Notes**: Security-relevant observations

| Package | Version | License | Native | Tier | Risk Notes |
|:---|:---|:---|:---|:---|:---|
| aiohappyeyeballs | 2.6.1 | PSF | N | 1 | — |
| aiohttp | 3.13.2 | Apache-2.0 | Y | 1 | HTTP client/server, C extensions |
| aiosignal | 1.4.0 | Apache-2.0 | N | 1 | — |
| aiosqlite | 0.21.0 | MIT | N | 1 | Async SQLite wrapper |
| annotated-types | 0.7.0 | MIT | N | 1 | — |
| anyio | 4.12.0 | MIT | N | 1 | — |
| atlassian-python-api | 4.0.7 | Apache-2.0 | N | 2 | Confluence/Jira API client |
| attrs | 25.4.0 | MIT | N | 1 | — |
| backoff | 2.2.1 | MIT | N | 1 | — |
| banks | 2.2.0 | MIT | N | 3 | Smaller templating project |
| beautifulsoup4 | 4.14.3 | MIT | N | 2 | HTML parsing |
| bm25s | 0.2.14 | MIT | N | 2 | BM25 retrieval, smaller project |
| certifi | 2025.11.12 | MPL-2.0 | N | 1 | CA certificate bundle |
| cffi | 2.0.0 | MIT | Y | 1 | C FFI bindings |
| charset-normalizer | 3.4.4 | MIT | N | 1 | — |
| click | 8.3.1 | BSD-3 | N | 1 | Pallets project |
| colorama | 0.4.6 | BSD-3 | N | 1 | — |
| coloredlogs | 15.0.1 | MIT | N | 1 | — |
| cryptography | 46.0.3 | Apache-2.0/BSD | Y | 1 | Rust + C extensions, high-value target but well-governed |
| cssselect2 | 0.8.0 | BSD-3 | N | 3 | Small project |
| dataclasses-json | 0.6.7 | MIT | N | 1 | — |
| defusedxml | 0.7.1 | PSF | N | 2 | Security-hardened XML parser |
| Deprecated | 1.2.18 | MIT | N | 1 | — |
| dirtyjson | 1.0.8 | MIT | N | 3 | Malformed JSON parser, single maintainer |
| distro | 1.9.0 | Apache-2.0 | N | 1 | — |
| docx2txt | 0.8 | MIT | N | 3 | Unmaintained since 2019 |
| fastembed | 0.7.4 | Apache-2.0 | N | 2 | Qdrant-backed, PyPI metadata incorrect |
| filelock | 3.20.0 | Unlicense | N | 1 | — |
| filetype | 1.2.0 | MIT | N | 3 | Single maintainer |
| FlashRank | 0.2.10 | Apache-2.0 | N | 2 | ONNX-based reranking |
| flatbuffers | 25.9.23 | Apache-2.0 | N | 1 | Google project |
| frozenlist | 1.8.0 | Apache-2.0 | Y | 1 | aio-libs |
| fsspec | 2025.12.0 | BSD-3 | N | 1 | — |
| greenlet | 3.3.0 | MIT | Y | 3 | Complex C extension for coroutines |
| griffe | 1.15.0 | ISC | N | 2 | — |
| h11 | 0.16.0 | MIT | N | 1 | — |
| hf-xet | 1.2.0 | Apache-2.0 | Y | 3 | New Rust extension from HF |
| httpcore | 1.0.9 | BSD-3 | N | 1 | — |
| httpx | 0.28.1 | BSD-3 | N | 1 | — |
| httpx-sse | 0.4.3 | MIT | N | 2 | — |
| huggingface-hub | 1.1.7 | Apache-2.0 | N | 1 | HF Inc. backed |
| humanfriendly | 10.0 | MIT | N | 1 | — |
| idna | 3.11 | BSD-3 | N | 1 | — |
| Jinja2 | 3.1.6 | BSD-3 | N | 1 | Pallets project |
| jira | 3.10.5 | BSD-2 | N | 2 | Jira API client |
| jiter | 0.12.0 | MIT | Y | 2 | Rust JSON parser (pydantic) |
| jmespath | 1.0.1 | MIT | N | 1 | — |
| joblib | 1.5.2 | BSD-3 | N | 1 | — |
| jsonschema | 4.25.1 | MIT | N | 1 | — |
| jsonschema-specifications | 2025.9.1 | MIT | N | 1 | — |
| llama-cloud | 0.1.35 | MIT | N | 2 | Pulls in posthog; unused by ChunkSilo |
| llama-cloud-services | 0.6.54 | MIT | N | 2 | Unused by ChunkSilo |
| llama-index | 0.14.10 | MIT | N | 2 | Core RAG framework |
| llama-index-cli | 0.5.3 | MIT | N | 2 | Unused by ChunkSilo |
| llama-index-core | 0.14.10 | MIT | N | 2 | Core dependency |
| llama-index-embeddings-fastembed | 0.5.0 | MIT | N | 2 | Direct dependency |
| llama-index-embeddings-openai | 0.5.1 | MIT | N | 2 | Unused by ChunkSilo |
| llama-index-indices-managed-llama-cloud | 0.9.4 | MIT | N | 2 | Unused by ChunkSilo |
| llama-index-instrumentation | 0.4.2 | MIT | N | 2 | — |
| llama-index-llms-openai | 0.6.10 | MIT | N | 2 | Unused by ChunkSilo |
| llama-index-readers-confluence | 0.6.0 | MIT | N | 2 | Confluence integration |
| llama-index-readers-file | 0.5.5 | MIT | N | 2 | File reading |
| llama-index-readers-llama-parse | 0.5.1 | MIT | N | 2 | Unused by ChunkSilo |
| llama-index-retrievers-bm25 | 0.6.5 | MIT | N | 2 | BM25 retrieval |
| llama-index-workflows | 2.11.5 | MIT | N | 2 | Transitive |
| llama-parse | 0.6.54 | MIT | N | 2 | Unused by ChunkSilo |
| loguru | 0.7.3 | MIT | N | 1 | — |
| lxml | 6.0.2 | BSD-3 | Y | 1 | XML/HTML parsing, C extension |
| markdown-it-py | 4.0.0 | MIT | N | 1 | — |
| markdownify | 1.2.2 | MIT | N | 2 | — |
| MarkupSafe | 3.0.3 | BSD-3 | Y | 1 | Pallets, C extension |
| marshmallow | 3.26.1 | MIT | N | 1 | — |
| mcp | 1.23.1 | MIT | N | 1 | Anthropic MCP protocol |
| mdurl | 0.1.2 | MIT | N | 1 | — |
| mmh3 | 5.2.0 | MIT | Y | 3 | C extension, MurmurHash3 |
| modelcontextprotocol | 1.0.1 | MIT | N | 1 | — |
| mpmath | 1.3.0 | BSD-3 | N | 1 | — |
| multidict | 6.7.0 | Apache-2.0 | Y | 1 | aio-libs, C extension |
| mypy_extensions | 1.1.0 | MIT | N | 1 | — |
| nest-asyncio | 1.6.0 | BSD-2 | N | 1 | — |
| networkx | 3.6 | BSD-3 | N | 1 | — |
| nltk | 3.9.2 | Apache-2.0 | N | 2 | Natural language toolkit |
| numpy | 2.3.5 | BSD-3 | Y | 1 | NumPy Foundation |
| oauthlib | 3.3.1 | BSD-3 | N | 1 | — |
| onnxruntime | 1.23.2 | MIT | Y | 1 | Microsoft, C++/CUDA extensions |
| openai | 2.9.0 | Apache-2.0 | N | 2 | Unused by ChunkSilo |
| packaging | 25.0 | Apache-2.0/BSD | N | 1 | — |
| pandas | 2.2.3 | BSD-3 | Y | 1 | NumPy Foundation |
| pdf2image | 1.17.0 | MIT | N | 2 | — |
| pillow | 10.3.0 | HPND | Y | 1 | Image processing, C extensions. CVE-prone but limited attack surface in our use. |
| platformdirs | 4.5.0 | MIT | N | 1 | — |
| posthog | 7.0.1 | MIT | N | 2 | **Telemetry SDK — must be disabled via env vars** |
| propcache | 0.4.1 | Apache-2.0 | Y | 1 | aio-libs |
| protobuf | 6.33.1 | BSD-3 | Y | 1 | Google |
| py_rust_stemmers | 0.1.5 | MIT | Y | 3 | Rust extension, single maintainer |
| pycparser | 2.23 | BSD-3 | N | 1 | — |
| pydantic | 2.12.5 | MIT | N | 1 | — |
| pydantic-settings | 2.12.0 | MIT | N | 1 | — |
| pydantic_core | 2.41.5 | MIT | Y | 1 | Rust extension, Pydantic Inc. |
| Pygments | 2.19.2 | BSD-2 | N | 1 | — |
| PyJWT | 2.10.1 | MIT | N | 1 | — |
| pypdf | 6.4.0 | BSD-3 | N | 1 | PDF parsing |
| pytesseract | 0.3.13 | Apache-2.0 | N | 2 | OCR wrapper (requires system Tesseract) |
| python-dateutil | 2.9.0.post0 | Apache-2.0/BSD | N | 1 | — |
| python-docx | 1.2.0 | MIT | N | 1 | — |
| python-dotenv | 1.2.1 | BSD-3 | N | 1 | — |
| python-multipart | 0.0.20 | Apache-2.0 | N | 1 | — |
| pytz | 2025.2 | MIT | N | 1 | — |
| PyYAML | 6.0.3 | MIT | Y | 1 | C extension (libyaml) |
| referencing | 0.37.0 | MIT | N | 1 | — |
| regex | 2025.11.3 | Apache-2.0 | Y | 1 | C extension |
| reportlab | 4.4.9 | BSD-3 | Y | 2 | PDF generation, C extensions |
| requests | 2.32.5 | Apache-2.0 | N | 1 | — |
| requests-oauthlib | 2.0.0 | ISC | N | 1 | — |
| requests-toolbelt | 1.0.0 | Apache-2.0 | N | 1 | — |
| retrying | 1.4.2 | Apache-2.0 | N | 2 | — |
| rich | 14.2.0 | MIT | N | 1 | — |
| rpds-py | 0.30.0 | MIT | Y | 1 | Rust extension |
| shellingham | 1.5.4 | ISC | N | 1 | — |
| six | 1.17.0 | MIT | N | 1 | — |
| sniffio | 1.3.1 | Apache-2.0/MIT | N | 1 | — |
| soupsieve | 2.8 | MIT | N | 1 | — |
| SQLAlchemy | 2.0.44 | MIT | Y | 1 | C extensions optional |
| sse-starlette | 3.0.3 | BSD-3 | N | 2 | — |
| starlette | 0.50.0 | BSD-3 | N | 1 | — |
| striprtf | 0.0.26 | BSD-3 | N | 3 | Small project, single maintainer |
| svglib | 1.5.1 | LGPL-3.0 | N | 2 | **Weak copyleft — dynamic import only, documented in NOTICE** |
| sympy | 1.14.0 | BSD-3 | N | 1 | — |
| tenacity | 9.1.2 | Apache-2.0 | N | 1 | — |
| tiktoken | 0.12.0 | MIT | Y | 1 | OpenAI tokenizer, Rust extension |
| tinycss2 | 1.5.1 | BSD-3 | N | 3 | Small project |
| tokenizers | 0.22.1 | Apache-2.0 | Y | 1 | HF Inc., Rust extension |
| tqdm | 4.67.1 | MPL-2.0/MIT | N | 1 | — |
| typer-slim | 0.20.0 | MIT | N | 1 | — |
| typing-inspect | 0.9.0 | MIT | N | 1 | — |
| typing-inspection | 0.4.2 | MIT | N | 1 | — |
| typing_extensions | 4.15.0 | PSF | N | 1 | — |
| tzdata | 2025.2 | Apache-2.0 | N | 1 | — |
| urllib3 | 2.5.0 | MIT | N | 1 | — |
| uvicorn | 0.38.0 | BSD-3 | N | 1 | — |
| webencodings | 0.5.1 | BSD-3 | N | 1 | — |
| wrapt | 1.17.3 | BSD-2 | Y | 1 | C extension |
| xlrd | 2.0.2 | BSD-3 | N | 3 | Maintenance mode |
| yarl | 1.22.0 | Apache-2.0 | Y | 1 | aio-libs, C extension |

**Bundled ML Models:**

| Model | Format | License | Source |
|:---|:---|:---|:---|
| BAAI/bge-small-en-v1.5 | ONNX (SafeTensors source) | MIT | HuggingFace |
| cross-encoder/ms-marco-MiniLM-L-12-v2 | ONNX | Apache-2.0 | HuggingFace |

Both models use the ONNX format, which is a declarative computation graph — it does not support arbitrary code execution during loading. This is safer than pickle-based model formats (e.g., PyTorch `.pt` files).
