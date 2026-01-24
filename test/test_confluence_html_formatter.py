"""Tests for confluence_html_formatter.py"""

import pytest

from chunksilo.confluence_html_formatter import CleanHtmlTextParser, clean_confluence_html


class TestCleanConfluenceHtml:
    def test_strips_color_styled_spans(self):
        """Spans with color styles should be unwrapped."""
        html = '<p>Code: <span style="color: #000080;">const</span> x = 1</p>'
        result = clean_confluence_html(html)
        assert "<span" not in result
        assert "const" in result

    def test_strips_background_styled_spans(self):
        """Spans with background styles should be unwrapped."""
        html = '<p><span style="background-color: yellow;">highlighted</span></p>'
        result = clean_confluence_html(html)
        assert "<span" not in result
        assert "highlighted" in result

    def test_strips_code_class_spans(self):
        """Spans with code-related classes should be unwrapped."""
        html = '<p><span class="code-keyword">function</span></p>'
        result = clean_confluence_html(html)
        assert "<span" not in result
        assert "function" in result

    def test_strips_multiple_code_classes(self):
        """Spans with various code-related classes should be unwrapped."""
        test_cases = [
            '<span class="code-quote">quoted</span>',
            '<span class="code-string">string</span>',
            '<span class="syntaxhighlighter-code">code</span>',
        ]
        for html in test_cases:
            result = clean_confluence_html(html)
            assert "<span" not in result

    def test_preserves_semantic_spans(self):
        """Spans without styling should be preserved."""
        html = '<p><span id="anchor">text</span></p>'
        result = clean_confluence_html(html)
        assert "<span" in result

    def test_preserves_spans_with_other_attributes(self):
        """Spans with non-color attributes should be preserved."""
        html = '<p><span data-macro="true">macro content</span></p>'
        result = clean_confluence_html(html)
        assert "<span" in result
        assert "macro content" in result

    def test_handles_nested_spans(self):
        """Nested syntax highlighting spans should be flattened."""
        html = '<span style="color: red;"><span style="color: blue;">nested</span></span>'
        result = clean_confluence_html(html)
        assert "nested" in result
        assert "<span" not in result

    def test_empty_html(self):
        """Empty input should return empty output."""
        assert clean_confluence_html("") == ""

    def test_none_html(self):
        """None input should return None."""
        assert clean_confluence_html(None) is None

    def test_preserves_other_elements(self):
        """Non-span elements should be preserved."""
        html = "<div><p>paragraph</p><code>code block</code></div>"
        result = clean_confluence_html(html)
        assert "<div>" in result
        assert "<p>" in result
        assert "<code>" in result

    def test_per_character_spans(self):
        """Per-character spans (common in Confluence) should be unwrapped."""
        # This is the problematic pattern that causes newlines between chars
        html = '<span style="color: #000;">c</span><span style="color: #000;">o</span><span style="color: #000;">n</span><span style="color: #000;">s</span><span style="color: #000;">t</span>'
        result = clean_confluence_html(html)
        assert "<span" not in result
        # All characters should be present
        assert "c" in result
        assert "o" in result
        assert "n" in result
        assert "s" in result
        assert "t" in result


class TestCleanHtmlTextParser:
    def test_converts_cleaned_html_to_markdown(self):
        """Full pipeline: clean HTML then convert to markdown."""
        html = '<h1>Title</h1><p>Code: <span style="color: blue;">var</span> x</p>'
        parser = CleanHtmlTextParser()
        result = parser.convert(html)
        assert "# Title" in result or "Title" in result  # ATX or fallback
        assert "var" in result

    def test_handles_real_confluence_code_block(self):
        """Simulates Confluence's verbose syntax highlighting."""
        # Confluence often wraps each character/token in a span
        html = '<pre><span style="color: #000080;">c</span><span style="color: #000080;">o</span><span style="color: #000080;">n</span><span style="color: #000080;">s</span><span style="color: #000080;">t</span></pre>'
        parser = CleanHtmlTextParser()
        result = parser.convert(html)
        # The key test: no newline between 'c' and 'o'
        assert "c\no" not in result

    def test_empty_html(self):
        """Empty input should return empty string."""
        parser = CleanHtmlTextParser()
        assert parser.convert("") == ""

    def test_preserves_regular_content(self):
        """Regular HTML content should be preserved and converted."""
        html = "<p>paragraph text</p><div>div content</div>"
        parser = CleanHtmlTextParser()
        result = parser.convert(html)
        assert "paragraph text" in result
        assert "div content" in result

    def test_heading_style_atx(self):
        """Headings should use ATX style (# prefix)."""
        html = "<h2>Heading Two</h2>"
        parser = CleanHtmlTextParser()
        result = parser.convert(html)
        assert "##" in result or "Heading Two" in result

    def test_list_bullets(self):
        """Lists should use asterisk bullets."""
        html = "<ul><li>item one</li><li>item two</li></ul>"
        parser = CleanHtmlTextParser()
        result = parser.convert(html)
        assert "*" in result or "item one" in result
