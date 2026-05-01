"""Reddit search via unauthenticated JSON API."""
from __future__ import annotations

import logging

import aiohttp

from localbot.config import cfg

log = logging.getLogger(__name__)


async def reddit_search(query: str, subreddit: str | None = None) -> str:
    if subreddit:
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {"q": query, "restrict_sr": "1", "sort": "relevance", "limit": cfg.search_result_count}
    else:
        url = "https://www.reddit.com/search.json"
        params = {"q": query, "sort": "relevance", "limit": cfg.search_result_count}

    headers = {"User-Agent": "LocalBot/0.1"}
    async with aiohttp.ClientSession() as session:
        async with session.get(
            url,
            params=params,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=cfg.tool_timeout_seconds),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()

    posts = data.get("data", {}).get("children", [])
    if not posts:
        return "No Reddit results found."

    lines = []
    for i, p in enumerate(posts[: cfg.search_result_count], 1):
        d = p["data"]
        lines.append(
            f"{i}. **{d.get('title', '')}** (r/{d.get('subreddit', '')})\n"
            f"   https://reddit.com{d.get('permalink', '')}\n"
            f"   ⬆ {d.get('score', 0)} | {d.get('num_comments', 0)} comments"
        )
    return "\n\n".join(lines)
