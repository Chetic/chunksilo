"""Text processing utilities."""

import re
from typing import List


def preprocess_query(query: str) -> str:
    """Preprocess queries with basic normalization.

    Techniques applied:
    - Normalize whitespace
    - Remove trailing punctuation that might interfere with matching

    Returns the original query if preprocessing results in an empty string.

    Args:
        query: The raw query string

    Returns:
        Preprocessed query string
    """
    if not query or not query.strip():
        return query

    original_query = query

    # Normalize whitespace (collapse multiple spaces)
    query = " ".join(query.split())

    # Remove trailing punctuation that might interfere with matching
    query = query.rstrip(".,!?;")

    # If preprocessing resulted in an empty string, return original query
    processed = query.strip()
    return processed if processed else original_query


def tokenize_filename(filename: str) -> List[str]:
    """Tokenize a filename for BM25 indexing.

    Splits on delimiters (underscore, hyphen, dot, space) and camelCase.

    Examples:
        'cpp_styleguide.md' -> ['cpp', 'styleguide', 'md']
        'API-Reference-v2.pdf' -> ['api', 'reference', 'v2', 'pdf']
        'CamelCaseDoc.docx' -> ['camel', 'case', 'doc', 'docx']

    Args:
        filename: The filename to tokenize

    Returns:
        List of lowercase tokens
    """
    name_parts = filename.rsplit(".", 1)
    base_name = name_parts[0]
    extension = name_parts[1] if len(name_parts) > 1 else ""

    # Split on explicit delimiters
    parts = re.split(r"[_\-\.\s]+", base_name)

    # Split camelCase within each part
    tokens = []
    for part in parts:
        camel_split = re.sub(r"([a-z])([A-Z])", r"\1 \2", part).split()
        tokens.extend(t.lower() for t in camel_split if t)

    # Add extension as a token
    if extension:
        tokens.append(extension.lower())

    return tokens


def compute_line_offsets(text: str) -> List[int]:
    """Compute character offset positions for each line start.

    Returns a list where line_offsets[i] is the character position where line i+1 starts.
    Line 1 starts at position 0 (implicit).

    Args:
        text: The text content

    Returns:
        List of character offsets for line starts
    """
    offsets = [0]  # Line 1 starts at position 0
    for i, char in enumerate(text):
        if char == "\n":
            offsets.append(i + 1)  # Next line starts after the newline
    return offsets
