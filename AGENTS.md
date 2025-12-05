# Agent Guidelines

- Use conventional commit messages **without a scope** (e.g., `feat: add progress bar`, `chore: update docs`).
- Keep future commits descriptive so release notes remain accurate.
- **Test files must not affect the release package**: All test-related files are in the `test/` directory and are excluded from the release zip file. The release package is a standalone, offline-ready MCP server that users can unpack and run without any test dependencies or online connections.
