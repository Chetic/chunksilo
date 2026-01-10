"""Heading extraction from various document formats."""

import logging
import re
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def extract_markdown_headings(text: str) -> List[dict]:
    """Extract heading hierarchy from Markdown text using ATX-style syntax.

    Parses # Heading syntax and returns list of dicts with text, position, level.
    Handles ATX-style headings (# Heading) but not Setext (underlined).

    Args:
        text: Markdown text content

    Returns:
        List of dicts with keys: text (str), position (int), level (int)
    """
    headings = []
    # Match ATX-style headings: line start, 1-6 #s, space, text
    pattern = re.compile(r"^(#{1,6})\s+(.+?)$", re.MULTILINE)

    # Find all code block ranges to skip headings inside them
    code_blocks = []
    for match in re.finditer(r"```.*?```", text, flags=re.DOTALL):
        code_blocks.append((match.start(), match.end()))

    def is_in_code_block(pos):
        """Check if position is inside a code block."""
        return any(start <= pos < end for start, end in code_blocks)

    for match in pattern.finditer(text):
        # Skip headings inside code blocks
        if is_in_code_block(match.start()):
            continue

        level = len(match.group(1))
        heading_text = match.group(2).strip()
        position = match.start()

        if heading_text:
            headings.append({"text": heading_text, "position": position, "level": level})

    return headings


def extract_pdf_headings_from_outline(pdf_path: Path) -> List[dict]:
    """Extract headings from PDF outline/bookmarks (TOC).

    Returns list of dicts with text, position (estimated), level.
    Position is approximate based on cumulative page character counts.
    Falls back to empty list if PDF has no outline or extraction fails.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dicts with keys: text (str), position (int), level (int)
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        logger.warning("PyMuPDF not available, skipping PDF heading extraction")
        return []

    try:
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()  # Returns [[level, title, page_num], ...]

        if not toc:
            return []

        headings = []
        for item in toc:
            level, title, page_num = item[0], item[1], item[2]

            # Estimate position by accumulating text from previous pages
            position = 0
            for page_idx in range(page_num - 1):
                if page_idx < len(doc):
                    page = doc[page_idx]
                    position += len(page.get_text())

            headings.append({"text": title.strip(), "position": position, "level": level})

        doc.close()
        return headings

    except Exception as e:
        logger.warning(f"Failed to extract PDF outline from {pdf_path}: {e}")
        return []
