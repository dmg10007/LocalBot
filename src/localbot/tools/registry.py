"""Tool registry and dispatcher."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING

from localbot.config import cfg
from localbot.tools import search, reddit, time_tools
from localbot.tools.log_reader import read_logs
from localbot.tools.scheduler_tools import SCHEDULER_TOOL_SCHEMAS

if TYPE_CHECKING:
    from localbot.tools.scheduler_tools import SchedulerTools

log = logging.getLogger(__name__)

# Static schemas for search / time / diagnostics tools.
# Descriptions are intentionally narrow — small models over-call tools
# when descriptions are broad.
_STATIC_SCHEMAS: list[dict[str, Any]] = [
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
    {
        "type": "function",
        "function": {
            "name": "read_logs",
            "description": (
                "Read recent audit log entries to diagnose errors, failed scheduled "
                "jobs, timeouts, or unexpected bot behaviour. Only call this when "
                "the user explicitly asks to check logs, troubleshoot an issue, or "
                "find out why something went wrong."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "enum": ["DEBUG", "INFO", "WARNING", "ERROR"],
                        "description": "Filter by severity. Omit to return all levels.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of entries to return (default 50, max 200).",
                    },
                },
                "required": [],
            },
        },
    },
]


def build_tool_schemas(include_scheduler: bool = False) -> list[dict[str, Any]]:
    """Return the full list of tool schemas for a request.

    Args:
        include_scheduler: Whether to include schedule_job / cancel_job /
            list_jobs schemas. Pass True when a live SchedulerService is
            available for this request.
    """
    if include_scheduler:
        return _STATIC_SCHEMAS + SCHEDULER_TOOL_SCHEMAS
    return _STATIC_SCHEMAS


async def dispatch(
    tool_name: str,
    args: dict[str, Any],
    scheduler_tools: "SchedulerTools | None" = None,
    requesting_user_id: str = "",
) -> str:
    """Run a tool by name and return its string result.

    Scheduler tools are routed to *scheduler_tools* when provided; all
    other tools are handled inline here.

    Args:
        tool_name: Name of the tool to invoke.
        args: Parsed JSON arguments from the LLM tool call.
        scheduler_tools: Per-request scheduler tool instance, if available.
        requesting_user_id: Discord user ID of the requester.  Required
            for read_logs scope enforcement.
    """
    try:
        async with asyncio.timeout(cfg.tool_timeout_seconds):
            # Scheduler tools: delegate to the per-request SchedulerTools instance.
            if scheduler_tools is not None:
                result = await scheduler_tools.dispatch(tool_name, args)
                if result is not None:
                    return result

            # Static tools.
            if tool_name == "web_search":
                return await search.web_search(args["query"])
            elif tool_name == "reddit_search":
                return await reddit.reddit_search(
                    args["query"], args.get("subreddit")
                )
            elif tool_name == "get_current_time":
                return time_tools.get_current_time(args.get("timezone", "UTC"))
            elif tool_name == "read_logs":
                return read_logs(
                    requesting_user_id=requesting_user_id,
                    level=args.get("level"),
                    limit=int(args.get("limit", 50)),
                )
            else:
                return f"Unknown tool: {tool_name}"
    except asyncio.TimeoutError:
        log.warning("Tool %s timed out", tool_name)
        return f"Tool '{tool_name}' timed out after {cfg.tool_timeout_seconds}s."
    except Exception as exc:
        log.exception("Tool %s raised an error", tool_name)
        return f"Tool '{tool_name}' error: {exc}"
