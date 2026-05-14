"""Core request/tool loop."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, TYPE_CHECKING

from localbot.adapters.llamacpp_client import LlamaCppClient
from localbot.adapters.llamacpp_server import LlamaCppServer
from localbot.config import cfg
from localbot.storage.audit import log_event
from localbot.storage.history import append_message, get_history
from localbot.tools.registry import build_tool_schemas, dispatch
# Fix #4: moved from inside handle() — no circular import exists here.
from localbot.tools.scheduler_tools import SchedulerTools

if TYPE_CHECKING:
    from localbot.scheduler.service import SchedulerService

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a helpful, concise assistant. You have the following tools:

SEARCH & TIME:
- web_search(query) — search the web for current information
- reddit_search(query, subreddit?) — search Reddit posts
- get_current_time(timezone?) — get the current date and time

SCHEDULING:
- schedule_job(prompt, cron_expr) — create a recurring scheduled message
- cancel_job(job_id) — cancel a scheduled job by its ID
- list_jobs() — list all active scheduled jobs for the user

RULES:
1. When the user asks to search, look something up, or find current news —
   call the tool immediately. Do NOT describe what you will do.
2. When the user asks to be reminded or wants a recurring message —
   call schedule_job immediately. Convert their natural-language schedule
   into a 5-field cron expression (minute hour day month day_of_week).
   Examples:
     "every day at 8am"          → cron_expr="0 8 * * *"
     "every Monday at 9am"       → cron_expr="0 9 * * 1"
     "every weekday at 6pm"      → cron_expr="0 18 * * 1-5"
     "every hour"                → cron_expr="0 * * * *"
     "every 30 minutes"          → cron_expr="*/30 * * * *"
   The user's timezone (if set) is used by the server; always express
   times in the user's local timezone when converting.
3. NEVER confirm a job is scheduled unless schedule_job returned successfully.
   NEVER invent a job ID. Always relay the ID returned by the tool.
4. After receiving tool results, write a clear, concise summary.
5. For casual conversation or simple questions — respond directly.
6. Never call the same tool with the same arguments twice in one turn.
7. Keep responses concise and friendly.
"""

_SYSTEM_ECHO_MARKERS = (
    "you are a helpful, concise assistant",
    "rules:",
    "1. when the user asks to search",
    "call the tool immediately",
)

_CONVERSATIONAL = re.compile(
    r"(hi+|hello+|hey+|howdy|sup|what'?s up|how are you|how r u|"
    r"good (morning|evening|afternoon|night)|thanks?( you)?|thank you|"
    r"ok(ay)?|sure|cool|great|nice|awesome|sounds good|got it|"
    r"bye|goodbye|see ya|later|lol|haha|:?\.?[)|(|D])\.?\!?",
    re.IGNORECASE,
)

_SEARCH_INTENT = re.compile(
    r"\b(search|look up|lookup|find|news|latest|current|today|trending|top stories)",
    re.IGNORECASE,
)

_SCHEDULE_INTENT = re.compile(
    r"\b(remind|reminder|schedule|recurring|every (day|week|hour|morning|evening|night|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekday|weekend|"
    r"\d+ (minute|hour|day)s?)|at \d+(am|pm|:\d+))",
    re.IGNORECASE,
)

_CANCEL_INTENT = re.compile(
    r"\b(cancel|remove|delete|stop|unschedule)\b.*\b(job|reminder|schedule)",
    re.IGNORECASE,
)

_TOOL_RESULT_MAX_CHARS = 4000


def _needs_tools(
    message: str,
    history: list[dict[str, Any]],
    has_scheduler: bool = False,
) -> bool:
    """Return True if this turn should have tools available."""
    stripped = message.strip()
    if _SEARCH_INTENT.search(stripped):
        return True
    if has_scheduler and (
        _SCHEDULE_INTENT.search(stripped) or _CANCEL_INTENT.search(stripped)
    ):
        return True
    # Fix #7: check all three intents against the most recent assistant turn,
    # not just search — so follow-ups like "actually cancel that" get tools.
    for turn in reversed(history[-4:]):
        if turn.get("role") == "assistant":
            content = turn.get("content") or ""
            if _SEARCH_INTENT.search(content):
                return True
            if has_scheduler and (
                _SCHEDULE_INTENT.search(content) or _CANCEL_INTENT.search(content)
            ):
                return True
            break
    return not bool(_CONVERSATIONAL.fullmatch(stripped))


def _is_system_echo(content: str) -> bool:
    lower = content.lower().strip()
    return any(marker in lower[:300] for marker in _SYSTEM_ECHO_MARKERS)


class Agent:
    def __init__(
        self,
        server: LlamaCppServer,
        client: LlamaCppClient,
        scheduler: "SchedulerService | None" = None,
    ) -> None:
        self._server = server
        self._client = client
        self._scheduler = scheduler

    async def handle(self, user_id: str, user_message: str) -> str:
        await self._server.ensure_running()
        # Fix #10: only call wait_until_ready when the client is not yet ready;
        # avoids an unnecessary /health HTTP probe on every message once the
        # server is up. LlamaCppClient.is_ready is set after the first
        # successful health check.
        if not self._client.is_ready:
            await self._client.wait_until_ready(retries=10, delay=1.0)

        history = get_history(user_id)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": user_message},
        ]

        log_event("user_message", user_id=user_id, content=user_message)

        # Build per-request scheduler tools bound to this user_id.
        sched_tools = (
            SchedulerTools(self._scheduler, user_id)
            if self._scheduler is not None
            else None
        )
        has_scheduler = sched_tools is not None

        timeout_sentinel = "Sorry, your request took too long and was cancelled."
        reply = timeout_sentinel
        try:
            async with asyncio.timeout(cfg.request_deadline_seconds):
                if _needs_tools(user_message, history, has_scheduler=has_scheduler):
                    tools = build_tool_schemas(include_scheduler=has_scheduler)
                else:
                    tools = None
                reply = await self._run_loop(messages, tools, sched_tools)
        except asyncio.TimeoutError:
            pass

        log_event("assistant_reply", user_id=user_id, content=reply)
        if reply and reply != timeout_sentinel:
            append_message(user_id, "user", user_message)
            append_message(user_id, "assistant", reply)

        return reply

    async def _run_loop(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        sched_tools: Any = None,
    ) -> str:
        called: set[tuple[str, str]] = set()

        for iteration in range(cfg.max_tool_iterations + 1):
            response = await self._client.chat(messages, tools=tools)
            choice = response["choices"][0]
            msg = choice["message"]
            content = msg.get("content") or ""

            if not msg.get("tool_calls"):
                if _is_system_echo(content):
                    log.warning("Model echoed system prompt — forcing synthesis.")
                    synth = await self._client.chat(
                        messages + [{"role": "user", "content": "Please answer the question above directly and concisely."}],
                        tools=None,
                    )
                    return synth["choices"][0]["message"].get("content") or ""
                return content

            if iteration == cfg.max_tool_iterations:
                messages.append({"role": "assistant", "content": None, "tool_calls": msg["tool_calls"]})
                messages.append({
                    "role": "user",
                    "content": "Please summarise all the information you have gathered and give me a final answer.",
                })
                final = await self._client.chat(messages, tools=None)
                return final["choices"][0]["message"].get("content") or ""

            messages.append(msg)
            any_new = False

            for tc in msg["tool_calls"]:
                fn = tc["function"]
                name = fn["name"]
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                dedup_key = (name, json.dumps(args, sort_keys=True))
                if dedup_key in called:
                    log.warning("Duplicate tool call blocked: %s(%s)", name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "[already called — use the result from the previous call]",
                    })
                    continue

                called.add(dedup_key)
                any_new = True

                log.info("Tool call: %s(%s)", name, args)
                log_event("tool_call", tool=name, args=args)
                result = await dispatch(name, args, scheduler_tools=sched_tools)
                log_event("tool_result", tool=name, result=result[:500])

                if len(result) > _TOOL_RESULT_MAX_CHARS:
                    log.debug(
                        "Tool result from %s truncated from %d to %d chars",
                        name, len(result), _TOOL_RESULT_MAX_CHARS,
                    )
                    result = result[:_TOOL_RESULT_MAX_CHARS] + "\n\n[...truncated]"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

            if not any_new:
                messages.append({
                    "role": "user",
                    "content": "You have all the information needed. Please give your final answer now.",
                })
                final = await self._client.chat(messages, tools=None)
                return final["choices"][0]["message"].get("content") or ""

        return "I was unable to complete your request."
