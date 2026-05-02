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
You are a helpful, concise assistant running locally via llama.cpp.

You have access to tools for web search, Reddit search, and getting the current time.
Only call tools when the user EXPLICITLY asks for information that requires them.
For greetings, casual chat, or anything conversational — respond directly without calling any tools.
After receiving tool results, always summarise them into a clear, helpful reply for the user.
Never call the same tool with the same arguments twice in one conversation turn.
Keep responses concise and friendly.
"""

_CONVERSATIONAL = re.compile(
    r"^(hi+|hello+|hey+|howdy|sup|what'?s up|how are you|how r u|"
    r"good (morning|evening|afternoon|night)|thanks?( you)?|thank you|"
    r"ok(ay)?|sure|cool|great|nice|awesome|sounds good|got it|"
    r"bye|goodbye|see ya|later|lol|haha|yes|no|yep|nope|:\.?[)|(|D])\.?\!?$",
    re.IGNORECASE,
)


def _needs_tools(message: str) -> bool:
    return not bool(_CONVERSATIONAL.match(message.strip()))


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

        try:
            async with asyncio.timeout(cfg.request_deadline_seconds):
                tools = TOOL_SCHEMAS if _needs_tools(user_message) else None
                reply = await self._run_loop(messages, tools)
        except asyncio.TimeoutError:
            reply = "Sorry, your request took too long and was cancelled."

        if reply:
            append_message(user_id, "user", user_message)
            append_message(user_id, "assistant", reply)
        log_event("assistant_reply", user_id=user_id, content=reply)
        return reply

    async def _run_loop(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> str:
        """Tool loop. messages is local-only — tool turns are never written to history."""
        # Track (tool_name, args_json) pairs called this turn to prevent duplicate calls.
        called: set[tuple[str, str]] = set()

        for iteration in range(cfg.max_tool_iterations + 1):
            response = await self._client.chat(messages, tools=tools)
            choice = response["choices"][0]
            msg = choice["message"]

            if not msg.get("tool_calls"):
                return msg.get("content") or ""

            if iteration == cfg.max_tool_iterations:
                # Iteration budget exhausted — force a plain-text synthesis.
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
                    # Model is repeating a call it already made — inject a
                    # cached notice so it moves on instead of looping.
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

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

            if not any_new:
                # Every call in this batch was a duplicate — force synthesis now.
                messages.append({
                    "role": "user",
                    "content": "You have all the information needed. Please give your final answer now.",
                })
                final = await self._client.chat(messages, tools=None)
                return final["choices"][0]["message"].get("content") or ""

        return "I was unable to complete your request."
