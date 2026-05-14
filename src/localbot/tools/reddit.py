"""Reddit search via unauthenticated JSON API."""
from __future__ import annotations

import logging
import re
from typing import Any

import aiohttp

from localbot.config import cfg

log = logging.getLogger(__name__)

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


def _clean_subreddit(subreddit: str) -> str:
    """Strip leading r/ or /r/ if the model included it."""
    return re.sub(r"^/?r/", "", subreddit.strip())


async def reddit_search(query: str, subreddit: str | None = None) -> str:
    # Fix #13: use dict[str, Any] instead of bare dict.
    if subreddit:
        subreddit = _clean_subreddit(subreddit)
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params: dict[str, Any] = {"q": query, "restrict_sr": "1", "sort": "relevance", "limit": cfg.search_result_count}
    else:
        url = "https://www.reddit.com/search.json"
        params: dict[str, Any] = {"q": query, "sort": "relevance", "limit": cfg.search_result_count}

    headers = {"User-Agent": "LocalBot/0.1"}
    session = _get_session()
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
            f"   \u2b06 {d.get('score', 0)} | {d.get('num_comments', 0)} comments"
        )
    return "\n\n".join(lines)
