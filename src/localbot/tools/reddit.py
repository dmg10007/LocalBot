"""Reddit search via unauthenticated JSON API."""
from __future__ import annotations

import logging
import re

import aiohttp

from localbot.config import cfg

log = logging.getLogger(__name__)


def _clean_subreddit(subreddit: str) -> str:
    """Strip leading r/ or /r/ if the model included it."""
    return re.sub(r"^/?r/", "", subreddit.strip())


async def reddit_search(query: str, subreddit: str | None = None) -> str:
    if subreddit:
        subreddit = _clean_subreddit(subreddit)
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params: dict = {"q": query, "restrict_sr": "1", "sort": "relevance", "limit": cfg.search_result_count}
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
