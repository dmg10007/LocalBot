"""Core request/tool loop."""
from __future__ import annotations

import asyncio
import json
import logging
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

You have access to these tools:
- web_search(query): Search the web for current information, news, or facts.
- reddit_search(query, subreddit): Search Reddit posts.
- get_current_time(timezone): Get the current date and time.

Rules:
- For greetings or simple conversational messages like "hello", "how are you", etc., respond directly WITHOUT calling any tools.
- Only call tools when the user explicitly asks for information that requires external data.
- Never call web_search or reddit_search unprompted.
- Keep responses concise and friendly.
"""


class Agent:
    def __init__(self, server: LlamaCppServer, client: LlamaCppClient) -> None:
        self._server = server
        self._client = client

    async def handle(self, user_id: str, user_message: str) -> str:
        """Process a user message and return the assistant reply."""
        await self._server.ensure_running()
        await self._client.wait_until_ready(retries=10, delay=1.0)

        history = get_history(user_id)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": user_message},
        ]

        append_message(user_id, "user", user_message)
        log_event("user_message", user_id=user_id, content=user_message)

        try:
            async with asyncio.timeout(cfg.request_deadline_seconds):
                reply = await self._run_loop(messages)
        except asyncio.TimeoutError:
            reply = "Sorry, your request took too long and was cancelled."

        append_message(user_id, "assistant", reply)
        log_event("assistant_reply", user_id=user_id, content=reply)
        return reply

    async def _run_loop(self, messages: list[dict[str, Any]]) -> str:
        for iteration in range(cfg.max_tool_iterations + 1):
            response = await self._client.chat(messages, tools=TOOL_SCHEMAS)
            choice = response["choices"][0]
            msg = choice["message"]

            # No tool call — return the final text
            if not msg.get("tool_calls"):
                return msg.get("content") or ""

            if iteration == cfg.max_tool_iterations:
                # Exceeded iteration budget — force a plain reply
                messages.append({"role": "assistant", "content": None, "tool_calls": msg["tool_calls"]})
                messages.append({
                    "role": "user",
                    "content": "Please provide your final answer based on the information gathered.",
                })
                final = await self._client.chat(messages, tools=None)
                return final["choices"][0]["message"].get("content") or ""

            # Execute all tool calls
            messages.append(msg)
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                name = fn["name"]
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                log.info("Tool call: %s(%s)", name, args)
                log_event("tool_call", tool=name, args=args)
                result = await dispatch(name, args)
                log_event("tool_result", tool=name, result=result[:500])

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

        return "I was unable to complete your request."
