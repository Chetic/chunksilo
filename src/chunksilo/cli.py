#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CLI entry point for the chunksilo command.

Usage:
    chunksilo "query text" [--date-from YYYY-MM-DD] [--date-to YYYY-MM-DD] [--config PATH] [--json]
    chunksilo --build-index [--config PATH]
    chunksilo --download-models [--config PATH]
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path


def main():
    """Entry point for the `chunksilo` command."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(
        prog="chunksilo",
        description="Search indexed documents using ChunkSilo",
        epilog=(
            "config file search order (first found wins):\n"
            "  1. --config PATH argument\n"
            "  2. CHUNKSILO_CONFIG environment variable\n"
            "  3. ./config.yaml\n"
            "  4. ~/.config/chunksilo/config.yaml\n"
            "  If none found, built-in defaults are used."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", nargs="?", default=None, help="Search query text")
    parser.add_argument("--date-from", help="Start date filter (YYYY-MM-DD, inclusive)")
    parser.add_argument("--date-to", help="End date filter (YYYY-MM-DD, inclusive)")
    parser.add_argument("--config", help="Path to config.yaml (overrides auto-discovery)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show diagnostic messages (model loading, search stats)")
    parser.add_argument("--build-index", action="store_true",
                        help="Build or update the search index, then exit")
    parser.add_argument("--download-models", action="store_true",
                        help="Download required ML models, then exit")
    parser.add_argument("--dump-defaults", action="store_true",
                        help="Print all default configuration values as YAML, then exit")
    parser.add_argument("--list-files", action="store_true",
                        help="List all indexed file paths, then exit")

    args = parser.parse_args()

    log_level = logging.WARNING if (args.build_index and not args.verbose) else (
        logging.INFO if (args.verbose or args.download_models) else logging.WARNING
    )
    logging.basicConfig(level=log_level, format="%(message)s", stream=sys.stderr)

    if args.dump_defaults:
        from .cfgload import _DEFAULTS
        import yaml
        yaml.dump(_DEFAULTS, sys.stdout, default_flow_style=False, sort_keys=False)
        return

    config_path = Path(args.config) if args.config else None

    if args.list_files:
        from .cfgload import load_config
        from .index import IngestionState

        cfg = load_config(config_path)
        state_db = Path(cfg["storage"]["storage_dir"]) / "ingestion_state.db"
        if not state_db.exists():
            print("No index found. Run chunksilo --build-index first.", file=sys.stderr)
            sys.exit(1)
        paths = sorted(IngestionState(state_db).get_all_files().keys())
        if args.json:
            print(json.dumps(paths, indent=2))
        else:
            for p in paths:
                print(p)
        return

    if args.build_index or args.download_models:
        # Suppress 3rd-party native output when IndexingUI owns the terminal
        if not args.verbose:
            os.environ["ORT_LOG_LEVEL"] = "ERROR"
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

        from .index import build_index

        build_index(
            download_only=args.download_models,
            config_path=config_path,
            verbose=args.verbose,
        )
        return

    if not args.query:
        parser.error("query is required (or use --build-index / --list-files / --download-models)")

    from .search import run_search

    result = run_search(
        query=args.query,
        date_from=args.date_from,
        date_to=args.date_to,
        config_path=config_path,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    # ANSI colors (disabled when stdout is not a TTY)
    _tty = sys.stdout.isatty()
    RESET = "\033[0m" if _tty else ""
    BOLD = "\033[1m" if _tty else ""
    DIM = "\033[2m" if _tty else ""
    CYAN = "\033[36m" if _tty else ""
    YELLOW = "\033[33m" if _tty else ""
    RED = "\033[31m" if _tty else ""
    BOLD_CYAN = "\033[1;36m" if _tty else ""

    # Check for errors
    if result.get("error"):
        print(f"{RED}Error: {result['error']}{RESET}", file=sys.stderr)
        sys.exit(1)

    # Human-readable output
    matched_files = result.get("matched_files", [])
    chunks = result.get("chunks", [])

    if matched_files:
        print(f"\n{BOLD}Matched files ({len(matched_files)}):{RESET}")
        for f in matched_files:
            print(f"  {CYAN}{f.get('uri', 'unknown')}{RESET}  {DIM}(score: {f.get('score', 0):.4f}){RESET}")

    if not chunks:
        print(f"\n{YELLOW}No results found.{RESET}")
        return

    print(f"\n{BOLD}Results ({len(chunks)}):{RESET}\n")

    for i, chunk in enumerate(chunks, 1):
        loc = chunk.get("location", {})
        uri = loc.get("uri") or "unknown"
        heading = " > ".join(loc.get("heading_path") or [])
        score = chunk.get("score", 0)

        print(f"{BOLD_CYAN}[{i}]{RESET} {CYAN}{uri}{RESET}")
        if heading:
            print(f"    {YELLOW}Heading: {heading}{RESET}")
        if loc.get("page"):
            print(f"    {DIM}Page: {loc['page']}{RESET}")
        if loc.get("line"):
            print(f"    {DIM}Line: {loc['line']}{RESET}")
        print(f"    {DIM}Score: {score:.3f}{RESET}")

        text = chunk.get("text", "")
        preview = text[:200].replace("\n", " ")
        if len(text) > 200:
            preview += "..."
        print(f"    {preview}")
        print()

    retrieval_time = result.get("retrieval_time", "")
    if retrieval_time:
        print(f"{DIM}({retrieval_time}){RESET}")
