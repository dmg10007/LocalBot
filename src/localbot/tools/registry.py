"""Tool registry and dispatcher."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from localbot.config import cfg
from localbot.tools import search, reddit, time_tools

log = logging.getLogger(__name__)

# OpenAI-style tool schemas sent to the model.
# Descriptions are intentionally narrow — small models over-call tools
# when descriptions are broad. "Use for X" should mean "only use for X".
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web. Only call this when the user explicitly asks to "
                "search, look something up, or requests current news or facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reddit_search",
            "description": (
                "Search Reddit posts. Only call this when the user explicitly asks "
                "to search Reddit or find Reddit discussions on a topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "subreddit": {
                        "type": "string",
                        "description": "Subreddit name without r/ prefix, e.g. 'worldnews'",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Get the current date and time. Only call this when the user "
                "explicitly asks for the time or date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone name, e.g. America/New_York. Defaults to UTC.",
                    }
                },
                "required": [],
            },
        },
    },
]


async def dispatch(tool_name: str, args: dict[str, Any]) -> str:
    """Run a tool by name and return its string result."""
    try:
        async with asyncio.timeout(cfg.tool_timeout_seconds):
            if tool_name == "web_search":
                return await search.web_search(args["query"])
            elif tool_name == "reddit_search":
                return await reddit.reddit_search(
                    args["query"], args.get("subreddit")
                )
            elif tool_name == "get_current_time":
                return time_tools.get_current_time(args.get("timezone", "UTC"))
            else:
                return f"Unknown tool: {tool_name}"
    except asyncio.TimeoutError:
        log.warning("Tool %s timed out", tool_name)
        return f"Tool '{tool_name}' timed out after {cfg.tool_timeout_seconds}s."
    except Exception as exc:
        log.exception("Tool %s raised an error", tool_name)
        return f"Tool '{tool_name}' error: {exc}"
