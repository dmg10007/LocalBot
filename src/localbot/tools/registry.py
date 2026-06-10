"""Tool registry and dispatcher."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal, TYPE_CHECKING

from localbot.config import cfg
from localbot.tools import search, reddit, time_tools
from localbot.tools.log_reader import read_logs
from localbot.tools.scheduler_tools import SCHEDULER_TOOL_SCHEMAS
from localbot.tools.filesystem import (
    FILESYSTEM_TOOL_SCHEMAS,
    read_file,
    write_file,
    list_directory,
    apply_patch,
    search_in_files,
)
from localbot.tools.github_tools import (
    GITHUB_TOOL_SCHEMAS,
    github_read_file,
    github_list_directory,
    github_create_branch,
    github_commit_files,
    github_create_pull_request,
    github_list_pull_requests,
)

if TYPE_CHECKING:
    from localbot.tools.scheduler_tools import SchedulerTools

log = logging.getLogger(__name__)

WorkspaceMode = Literal["local", "github", "both", None]

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


def build_tool_schemas(
    include_scheduler: bool = False,
    workspace_mode: WorkspaceMode = None,
) -> list[dict[str, Any]]:
    """Return the full list of tool schemas for a request.

    Args:
        include_scheduler: Whether to include schedule_job / cancel_job /
            list_jobs schemas.
        workspace_mode: 'local' adds filesystem tools; 'github' adds GitHub
            tools; 'both' adds both sets; None adds neither.
    """
    schemas = list(_STATIC_SCHEMAS)
    if include_scheduler:
        schemas += SCHEDULER_TOOL_SCHEMAS
    if workspace_mode in ("local", "both") and cfg.sandbox_root:
        schemas += FILESYSTEM_TOOL_SCHEMAS
    if workspace_mode in ("github", "both") and cfg.github_token:
        schemas += GITHUB_TOOL_SCHEMAS
    return schemas


async def dispatch(
    tool_name: str,
    args: dict[str, Any],
    scheduler_tools: "SchedulerTools | None" = None,
    requesting_user_id: str = "",
) -> str:
    """Run a tool by name and return its string result.

    Scheduler tools are routed to *scheduler_tools* when provided; all
    other tools are handled inline here.
    """
    try:
        async with asyncio.timeout(cfg.tool_timeout_seconds):
            # Scheduler tools.
            if scheduler_tools is not None:
                result = await scheduler_tools.dispatch(tool_name, args)
                if result is not None:
                    return result

            # Static / web tools.
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

            # Local filesystem tools.
            elif tool_name == "read_file":
                return read_file(args["path"])
            elif tool_name == "write_file":
                return write_file(args["path"], args["content"])
            elif tool_name == "list_directory":
                return list_directory(args.get("path", "."))
            elif tool_name == "apply_patch":
                return apply_patch(args["path"], args["patch"])
            elif tool_name == "search_in_files":
                return search_in_files(
                    args["pattern"],
                    path=args.get("path", "."),
                    file_glob=args.get("file_glob", "*"),
                )

            # GitHub tools.
            elif tool_name == "github_read_file":
                return await github_read_file(
                    args["owner"], args["repo"], args["path"],
                    ref=args.get("ref", "HEAD"),
                )
            elif tool_name == "github_list_directory":
                return await github_list_directory(
                    args["owner"], args["repo"],
                    path=args.get("path", ""),
                    ref=args.get("ref", "HEAD"),
                )
            elif tool_name == "github_create_branch":
                return await github_create_branch(
                    args["owner"], args["repo"], args["branch"],
                    from_branch=args.get("from_branch", "main"),
                )
            elif tool_name == "github_commit_files":
                return await github_commit_files(
                    args["owner"], args["repo"], args["branch"],
                    args["message"], args["files"],
                )
            elif tool_name == "github_create_pull_request":
                return await github_create_pull_request(
                    args["owner"], args["repo"], args["title"],
                    args["head"],
                    base=args.get("base", "main"),
                    body=args.get("body", ""),
                )
            elif tool_name == "github_list_pull_requests":
                return await github_list_pull_requests(
                    args["owner"], args["repo"],
                    state=args.get("state", "open"),
                )

            else:
                return f"Unknown tool: {tool_name}"
    except asyncio.TimeoutError:
        log.warning("Tool %s timed out", tool_name)
        return f"Tool '{tool_name}' timed out after {cfg.tool_timeout_seconds}s."
    except Exception as exc:
        log.exception("Tool %s raised an error", tool_name)
        return f"Tool '{tool_name}' error: {exc}"
