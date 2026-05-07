"""Async HTTP client for the llama-server OpenAI-compatible API.

A single shared aiohttp.ClientSession is created at first use and reused
for all subsequent requests, avoiding per-call TCP overhead.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import aiohttp

from localbot.config import cfg

log = logging.getLogger(__name__)

# Gemma end-of-turn token used for plain-text (non-tool) responses only.
# Do NOT include this in tool-call requests — it fires inside JSON payloads
# and truncates tool arguments, causing silent JSONDecodeErrors.
_PLAIN_STOP_TOKENS = [
    "<end_of_turn>",
    "<|eot_id|>",
    "<|end_of_text|>",
    "<|eom_id|>",
]

# Matches a full <think>...</think> block, including newlines.
# The (?:<think>)? handles the GLM/Gemma edge-case where the opening tag is
# injected by the chat template and absent from the raw model output.
_THINK_RE = re.compile(
    r"(?:<think>)?.*?</think>",
    re.DOTALL | re.IGNORECASE,
)


def strip_thinking(message: dict[str, Any]) -> str:
    """Extract the user-facing reply from a chat completion message dict.

    llama.cpp surfaces thinking-model output in one of two ways:
      1. A separate ``reasoning_content`` field (preferred — already split).
      2. Inside ``content`` as a ``<think>...</think>`` block before the reply.

    In both cases the thinking text is discarded and only the final answer
    is returned. The raw thinking block is logged at DEBUG level.
    """
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""

    if reasoning:
        log.debug("[thinking] %s", reasoning[:500])
        return content.strip()

    if "</think>" in content:
        think_match = _THINK_RE.match(content.lstrip())
        if think_match:
            log.debug("[thinking] %s", think_match.group(0)[:500])
        return _THINK_RE.sub("", content).strip()

    return content.strip()


class LlamaCppClient:
    def __init__(self) -> None:
        self._base = f"http://{cfg.llama_server_host}:{cfg.llama_server_port}"
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Call /v1/chat/completions and strip thinking blocks from the reply."""
        payload: dict[str, Any] = {
            "messages": messages,
            "stream": False,
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 2048,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
            # Do NOT set stop tokens when tools are enabled.
            # <end_of_turn> fires inside tool-call JSON and truncates arguments.
        else:
            payload["stop"] = _PLAIN_STOP_TOKENS

        session = self._get_session()
        resp = await session.post(
            f"{self._base}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=cfg.model_timeout_seconds),
        )

        # Raise on all errors — do NOT silently retry without tools on 500.
        # The old fallback caused the model to respond as if it had no tools,
        # telling users "I don't have web search access". Errors surface in
        # logs and are caught by the agent's asyncio.timeout wrapper.
        resp.raise_for_status()

        data: dict[str, Any] = await resp.json()

        # Strip <think>...</think> blocks from plain-text replies only.
        # Tool-call messages must not be mutated — their content is JSON.
        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            if not msg.get("tool_calls"):
                msg["content"] = strip_thinking(msg)

        return data

    async def wait_until_ready(self, retries: int = 20, delay: float = 1.5) -> None:
        session = self._get_session()
        for attempt in range(retries):
            try:
                async with session.get(
                    f"{self._base}/health",
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as r:
                    if r.status == 200:
                        log.info("llama-server is ready")
                        return
            except Exception:
                pass
            log.debug("Waiting for llama-server... (%d/%d)", attempt + 1, retries)
            await asyncio.sleep(delay)
        raise RuntimeError("llama-server did not become ready in time")
