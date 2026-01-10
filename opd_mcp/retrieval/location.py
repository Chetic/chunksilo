"""Location utilities for chunk position tracking."""

from typing import List, Optional, Tuple


def build_heading_path(
    headings: List[dict], char_start: Optional[int]
) -> Tuple[Optional[str], List[str]]:
    """Build a human-readable heading path for a given character position.

    Args:
        headings: List of heading dicts with 'text', 'position', 'level' keys
        char_start: Character position within the document

    Returns:
        Tuple of (current_heading_text, heading_path_list)
        e.g., ("Deployment", ["Architecture", "CI/CD", "Deployment"])
    """
    if not headings or char_start is None:
        return None, []

    # Find the index of the current heading
    current_idx = None
    for idx, heading in enumerate(headings):
        heading_pos = heading.get("position", 0)
        if heading_pos <= char_start:
            current_idx = idx
        else:
            break

    if current_idx is None:
        return None, []

    # Build path from all headings up to and including the current one
    path = [h.get("text", "") for h in headings[: current_idx + 1] if h.get("text")]
    current_heading_text = path[-1] if path else None
    return current_heading_text, path


def char_offset_to_line(
    char_offset: Optional[int], line_offsets: Optional[List[int]]
) -> Optional[int]:
    """Convert a character offset to a line number (1-indexed).

    Uses precomputed line offsets for efficient lookup.

    Args:
        char_offset: Character position in the document (0-indexed)
        line_offsets: List where line_offsets[i] is the char position where line i+1 starts

    Returns:
        Line number (1-indexed) or None if cannot be determined
    """
    if char_offset is None or not line_offsets:
        return None

    # Binary search to find the line containing char_offset
    left, right = 0, len(line_offsets) - 1
    while left < right:
        mid = (left + right + 1) // 2
        if line_offsets[mid] <= char_offset:
            left = mid
        else:
            right = mid - 1

    return left + 1  # Convert to 1-indexed line number
