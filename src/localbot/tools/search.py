"""Web search via Brave Search API with full-page content fetching."""
from __future__ import annotations

import asyncio
import itertools
import logging
import re
from typing import Any

import aiohttp
from bs4 import BeautifulSoup

from localbot.config import cfg

log = logging.getLogger(__name__)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Module-level session reused across calls to avoid per-request TCP overhead.
_session: aiohttp.ClientSession | None = None

# User-agent for page fetching. Some sites block the default aiohttp UA.
_FETCH_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Domains that reliably block scrapers or return useless content.
# Fix #9 (prior): "pdf" removed from this set — it was matching any URL
# containing the substring "pdf". PDF files are detected in _should_skip()
# via an endswith() check on the path instead.
_SKIP_DOMAINS = frozenset([
    "youtube.com", "youtu.be",
    "twitter.com", "x.com",
    "instagram.com", "facebook.com",
    "tiktok.com",
])


def _get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


async def close_session() -> None:
    """Close the shared session. Call this on bot shutdown."""
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None


def _should_skip(url: str) -> bool:
    """Return True for URLs we know won't yield useful scraped text."""
    lower = url.lower()
    # Detect raw PDF URLs by checking the path extension, not by substring
    # matching which incorrectly skips /pdf-guide/, pdfhost.io, etc.
    path = lower.split("?")[0]  # strip query string before checking extension
    if path.endswith(".pdf"):
        return True
    return any(domain in lower for domain in _SKIP_DOMAINS)


def _extract_text(html: str, max_chars: int) -> str:
    """Strip HTML and return clean readable text up to max_chars."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "form", "noscript", "iframe"]):
        tag.decompose()

    # Prefer <article> or <main> if available for higher signal-to-noise ratio
    body = soup.find("article") or soup.find("main") or soup.body or soup

    text = body.get_text(separator=" ", strip=True)  # type: ignore[union-attr]

    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text[:max_chars].strip()


async def _fetch_page(url: str) -> str | None:
    """Fetch a single URL and return extracted text, or None on failure."""
    if _should_skip(url):
        log.debug("Skipping unsupported URL: %s", url)
        return None

    session = _get_session()
    try:
        async with session.get(
            url,
            headers={"User-Agent": _FETCH_UA},
            timeout=aiohttp.ClientTimeout(total=cfg.search_fetch_timeout_seconds),
            allow_redirects=True,
        ) as resp:
            if resp.status != 200:
                log.debug("Page fetch %s returned HTTP %d", url, resp.status)
                return None
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                log.debug("Skipping non-HTML content-type at %s: %s", url, content_type)
                return None
            html = await resp.text(errors="replace")
        return _extract_text(html, cfg.search_fetch_chars)
    except Exception as exc:
        log.debug("Failed to fetch %s: %s", url, exc)
        return None


async def web_search(query: str) -> str:
    """Search the web and return titles, URLs, and extracted page content."""
    if not cfg.brave_api_key:
        return "Web search is disabled (BRAVE_API_KEY not set)."

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": cfg.brave_api_key,
    }
    params = {"q": query, "count": cfg.search_result_count}

    session = _get_session()
    async with session.get(
        BRAVE_SEARCH_URL,
        headers=headers,
        params=params,
        timeout=aiohttp.ClientTimeout(total=cfg.tool_timeout_seconds),
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()

    results: list[dict[str, Any]] = data.get("web", {}).get("results", [])
    if not results:
        return "No results found."

    # Fetch the top N pages concurrently.
    top = results[: cfg.search_result_count]
    fetch_targets = top[: cfg.search_fetch_count]
    page_texts = await asyncio.gather(
        *[_fetch_page(r.get("url", "")) for r in fetch_targets],
        return_exceptions=False,
    )

    # Fix #9: use zip_longest to pair results with their fetched text without
    # fragile manual index arithmetic. Results beyond fetch_targets get None.
    lines: list[str] = []
    for i, (r, page_content) in enumerate(
        itertools.zip_longest(top, page_texts, fillvalue=None), 1
    ):
        if r is None:
            break
        title = r.get("title", "")
        url = r.get("url", "")
        description = r.get("description", "")

        block = f"{i}. **{title}**\n   {url}"
        if page_content:
            block += f"\n\n   {page_content}"
        elif description:
            block += f"\n   {description}"

        lines.append(block)

    return "\n\n---\n\n".join(lines)
