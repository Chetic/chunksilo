#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CLI entry point for the search_docs command.

Usage:
    search_docs "query text" [--date-from YYYY-MM-DD] [--date-to YYYY-MM-DD] [--config PATH] [--json]
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    """Entry point for the `search_docs` command."""
    parser = argparse.ArgumentParser(
        prog="search_docs",
        description="Search indexed documents using ChunkSilo",
    )
    parser.add_argument("query", help="Search query text")
    parser.add_argument("--date-from", help="Start date filter (YYYY-MM-DD, inclusive)")
    parser.add_argument("--date-to", help="End date filter (YYYY-MM-DD, inclusive)")
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    from .search import run_search

    config_path = Path(args.config) if args.config else None
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
    if matched_files:
        print(f"\n--- Matched Files ({len(matched_files)}) ---")
        for f in matched_files:
            print(f"  {f.get('uri', 'unknown')}  (score: {f.get('score', 0):.4f})")

    chunks = result.get("chunks", [])
    retrieval_time = result.get("retrieval_time", "")
    print(f"\n--- Results ({len(chunks)}) --- [{retrieval_time}]\n")

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
        # Show first 200 chars of text, truncated
        preview = text[:200].replace("\n", " ")
        if len(text) > 200:
            preview += "..."
        print(f"    {preview}")
        print()
