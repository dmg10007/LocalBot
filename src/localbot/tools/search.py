"""Web search via Brave Search API."""
from __future__ import annotations

import logging

import aiohttp

from localbot.config import cfg

log = logging.getLogger(__name__)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Module-level session reused across calls to avoid per-request TCP overhead.
# aiohttp explicitly recommends against creating a new session per request.
_session: aiohttp.ClientSession | None = None


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


async def web_search(query: str) -> str:
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

    results = data.get("web", {}).get("results", [])
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results[: cfg.search_result_count], 1):
        lines.append(f"{i}. **{r.get('title', '')}**\n   {r.get('url', '')}\n   {r.get('description', '')}")
    return "\n\n".join(lines)
