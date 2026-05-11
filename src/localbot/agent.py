"""Core request/tool loop."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from localbot.adapters.llamacpp_client import LlamaCppClient
from localbot.adapters.llamacpp_server import LlamaCppServer
from localbot.config import cfg
from localbot.storage.audit import log_event
from localbot.storage.history import append_message, get_history
from localbot.tools.registry import TOOL_SCHEMAS, dispatch

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a helpful, concise assistant. You have three tools:
- web_search(query) — search the web for current information
- reddit_search(query, subreddit?) — search Reddit posts
- get_current_time(timezone?) — get the current date and time

RULES:
1. When the user asks you to search, look something up, or find current news —
   you MUST call the tool immediately. Do NOT describe what you will do.
   Do NOT say "I will search". Do NOT fabricate results or use placeholders.
   Just call the tool.
2. After receiving tool results, write a clear, concise summary with source links.
3. For casual conversation, greetings, or simple questions you can answer from
   knowledge — respond directly without calling any tools.
4. Never call the same tool with the same arguments twice in one turn.
5. Keep responses concise and friendly.
"""

# Fix #17: structural echo detection — markers are tied to the shape of the
# system prompt, not arbitrary phrases that silently rot when the prompt changes.
_SYSTEM_ECHO_MARKERS = (
    "you are a helpful, concise assistant",
    "rules:",
    "1. when the user asks you to search",
    "you must call the tool immediately",
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

# Fix #10: cap how many characters of a tool result are injected into the
# context window to prevent runaway search responses from exhausting model RAM.
_TOOL_RESULT_MAX_CHARS = 4000


def _needs_tools(message: str, history: list[dict[str, Any]]) -> bool:
    """Return True if this turn should have tools available."""
    stripped = message.strip()
    if _SEARCH_INTENT.search(stripped):
        return True
    for turn in reversed(history[-4:]):
        if turn.get("role") == "assistant":
            content = turn.get("content") or ""
            if _SEARCH_INTENT.search(content):
                return True
            break
    return not bool(_CONVERSATIONAL.fullmatch(stripped))


def _is_system_echo(content: str) -> bool:
    """Return True if the model echoed the system prompt as its reply.

    Fix #17: checks structural markers coupled to the prompt shape rather
    than hardcoded phrases that silently stop working if the prompt changes.
    """
    lower = content.lower().strip()
    return any(marker in lower[:300] for marker in _SYSTEM_ECHO_MARKERS)


class Agent:
    def __init__(self, server: LlamaCppServer, client: LlamaCppClient) -> None:
        self._server = server
        self._client = client

    async def handle(self, user_id: str, user_message: str) -> str:
        await self._server.ensure_running()
        await self._client.wait_until_ready(retries=10, delay=1.0)

        history = get_history(user_id)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": user_message},
        ]

        log_event("user_message", user_id=user_id, content=user_message)

        # Fix #2: define the timeout sentinel so we can distinguish it from a
        # real LLM reply when deciding whether to persist history.
        timeout_sentinel = "Sorry, your request took too long and was cancelled."
        reply = timeout_sentinel
        try:
            async with asyncio.timeout(cfg.request_deadline_seconds):
                tools = TOOL_SCHEMAS if _needs_tools(user_message, history) else None
                reply = await self._run_loop(messages, tools)
        except asyncio.TimeoutError:
            pass  # reply stays as timeout_sentinel

        # Fix #2: audit log is always written (including timeouts), but history
        # is only persisted for genuine LLM replies, not timeout messages.
        log_event("assistant_reply", user_id=user_id, content=reply)
        if reply and reply != timeout_sentinel:
            append_message(user_id, "user", user_message)
            append_message(user_id, "assistant", reply)

        return reply

    async def _run_loop(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> str:
        """Tool loop. messages is local-only — tool turns are never written to history."""
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
                result = await dispatch(name, args)
                log_event("tool_result", tool=name, result=result[:500])

                # Fix #10: cap tool result length before injecting into context.
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
