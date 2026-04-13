"""Tests for the fetch scraper HTML extraction logic."""

from maestro.scrapers.fetch import _collapse_whitespace, _extract_text_from_html


class TestExtractTextFromHtml:
    def test_simple_paragraph(self):
        html = "<html><body><p>Hello world</p></body></html>"
        result = _extract_text_from_html(html)
        assert "Hello world" in result

    def test_strips_scripts(self):
        html = "<html><body><p>Content</p><script>alert('xss')</script></body></html>"
        result = _extract_text_from_html(html)
        assert "Content" in result
        assert "alert" not in result

    def test_strips_style(self):
        html = "<html><body><p>Content</p><style>.foo{color:red}</style></body></html>"
        result = _extract_text_from_html(html)
        assert "Content" in result
        assert "color" not in result

    def test_strips_nav(self):
        html = "<html><body><nav>Navigation</nav><main><p>Main content</p></main></body></html>"
        result = _extract_text_from_html(html)
        assert "Main content" in result
        assert "Navigation" not in result

    def test_strips_footer(self):
        html = "<html><body><p>Content</p><footer>Footer text</footer></body></html>"
        result = _extract_text_from_html(html)
        assert "Content" in result
        assert "Footer text" not in result

    def test_strips_aria_hidden(self):
        html = '<html><body><p>Visible</p><div aria-hidden="true">Hidden</div></body></html>'
        result = _extract_text_from_html(html)
        assert "Visible" in result
        assert "Hidden" not in result

    def test_block_elements_produce_newlines(self):
        html = "<html><body><p>First</p><p>Second</p></body></html>"
        result = _extract_text_from_html(html)
        assert "First" in result
        assert "Second" in result
        # Block elements should create separation
        assert result.index("First") < result.index("Second")

    def test_empty_html(self):
        result = _extract_text_from_html("")
        assert result == ""

    def test_nested_content(self):
        html = "<html><body><div><span>Nested <strong>bold</strong> text</span></div></body></html>"
        result = _extract_text_from_html(html)
        assert "Nested bold text" in result


class TestCollapseWhitespace:
    def test_multiple_spaces(self):
        assert _collapse_whitespace("hello    world") == "hello world"

    def test_multiple_blank_lines(self):
        result = _collapse_whitespace("a\n\n\n\n\nb")
        assert result == "a\n\nb"

    def test_tabs_collapsed(self):
        assert _collapse_whitespace("hello\t\tworld") == "hello world"

    def test_preserves_single_newline(self):
        assert _collapse_whitespace("a\nb") == "a\nb"
