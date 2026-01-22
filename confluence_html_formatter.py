"""
Custom HTML formatter for Confluence content.

TEMPORARY FIX: Strips syntax highlighting <span> tags that cause issues
with markdownify. Remove this file when the upstream issue is fixed.

To remove this workaround:
1. Delete this file
2. Remove the import and patch_confluence_reader() call from chunksilo.py
"""

from bs4 import BeautifulSoup


def clean_confluence_html(html: str) -> str:
    """
    Pre-process Confluence HTML to remove problematic syntax highlighting spans.

    Confluence wraps code in many <span> tags for syntax highlighting that can
    cause markdownify to insert unwanted newlines between characters.

    Args:
        html: Raw HTML string from Confluence

    Returns:
        Cleaned HTML with syntax highlighting spans unwrapped
    """
    if not html:
        return html

    soup = BeautifulSoup(html, "html.parser")

    # Find and unwrap syntax highlighting spans
    # These are typically <span> tags with class attributes for highlighting
    # or inline style attributes for colors
    for span in soup.find_all("span"):
        # Check if this looks like a syntax highlighting span
        span_class = span.get("class", [])
        span_style = span.get("style", "")

        # Common patterns: spans with color styles, or code-related classes
        is_syntax_span = (
            "color" in span_style
            or "background" in span_style
            or any("code" in c for c in span_class if isinstance(c, str))
        )

        if is_syntax_span:
            span.unwrap()  # Replace span with its contents

    return str(soup)


class CleanHtmlTextParser:
    """
    Drop-in replacement for llama_index's HtmlTextParser that cleans
    syntax highlighting spans before conversion.
    """

    def __init__(self):
        try:
            from markdownify import markdownify  # noqa: F401
        except ImportError:
            raise ImportError(
                "`markdownify` package not found, please run `pip install markdownify`"
            )

    def convert(self, html: str) -> str:
        from markdownify import markdownify

        if not html:
            return ""

        # Clean the HTML first
        cleaned_html = clean_confluence_html(html)

        return markdownify(
            cleaned_html,
            heading_style="ATX",
            bullets="*",
            strip=["script", "style"],
        )


def patch_confluence_reader():
    """
    Monkey-patch the ConfluenceReader to use our clean HTML parser.
    Call this before creating ConfluenceReader instances.
    """
    try:
        import llama_index.readers.confluence.html_parser as html_parser_module

        html_parser_module.HtmlTextParser = CleanHtmlTextParser
    except ImportError:
        pass  # ConfluenceReader not installed
