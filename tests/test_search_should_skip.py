"""Tests for tools.search._should_skip."""
import pytest
from localbot.tools.search import _should_skip


@pytest.mark.parametrize("url", [
    "https://example.com/document.pdf",
    "https://example.com/report.PDF",
    "https://example.com/path/to/file.pdf",
    "https://youtube.com/watch?v=abc",
    "https://www.youtube.com/watch?v=abc",
    "https://youtu.be/abc",
    "https://twitter.com/user/status/123",
    "https://x.com/user/status/123",
    "https://instagram.com/p/abc",
    "https://facebook.com/post/123",
    "https://tiktok.com/@user/video/123",
])
def test_urls_that_should_be_skipped(url):
    assert _should_skip(url) is True


@pytest.mark.parametrize("url", [
    "https://example.com/pdf-guide",           # 'pdf' in path but not extension
    "https://pdfhost.io/some-document",        # 'pdf' in domain but not extension
    "https://example.com/guide.pdf.html",      # ends in .html, not .pdf
    "https://github.com/user/repo",
    "https://docs.python.org/3/library/",
    "https://reddit.com/r/python",
    "https://stackoverflow.com/questions/123",
])
def test_urls_that_should_not_be_skipped(url):
    assert _should_skip(url) is False
