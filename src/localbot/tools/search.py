"""Web search via Brave Search API."""
from __future__ import annotations

import logging

import aiohttp

from localbot.config import cfg

log = logging.getLogger(__name__)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


async def web_search(query: str) -> str:
    if not cfg.brave_api_key:
        return "Web search is disabled (BRAVE_API_KEY not set)."

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": cfg.brave_api_key,
    }
    params = {"q": query, "count": cfg.search_result_count}

    async with aiohttp.ClientSession() as session:
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
