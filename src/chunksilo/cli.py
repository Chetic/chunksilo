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

    args = parser.parse_args()

    log_level = logging.INFO if args.verbose or args.build_index or args.download_models else logging.WARNING
    logging.basicConfig(level=log_level, format="%(message)s", stream=sys.stderr)

    config_path = Path(args.config) if args.config else None

    if args.build_index or args.download_models:
        from .index import build_index

        build_index(
            download_only=args.download_models,
            config_path=config_path,
        )
        return

    if not args.query:
        parser.error("query is required (or use --build-index / --download-models)")

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

    # Check for errors
    if result.get("error"):
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    # Human-readable output
    matched_files = result.get("matched_files", [])
    chunks = result.get("chunks", [])

    if matched_files:
        print(f"\nMatched files ({len(matched_files)}):")
        for f in matched_files:
            print(f"  {f.get('uri', 'unknown')}  (score: {f.get('score', 0):.4f})")

    if not chunks:
        print("\nNo results found.")
        return

    print(f"\nResults ({len(chunks)}):\n")

    for i, chunk in enumerate(chunks, 1):
        loc = chunk.get("location", {})
        uri = loc.get("uri") or "unknown"
        heading = " > ".join(loc.get("heading_path") or [])
        score = chunk.get("score", 0)

        print(f"[{i}] {uri}")
        if heading:
            print(f"    Heading: {heading}")
        if loc.get("page"):
            print(f"    Page: {loc['page']}")
        if loc.get("line"):
            print(f"    Line: {loc['line']}")
        print(f"    Score: {score:.3f}")

        text = chunk.get("text", "")
        preview = text[:200].replace("\n", " ")
        if len(text) > 200:
            preview += "..."
        print(f"    {preview}")
        print()

    retrieval_time = result.get("retrieval_time", "")
    if retrieval_time:
        print(f"({retrieval_time})")
