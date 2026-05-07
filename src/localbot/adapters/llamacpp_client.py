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

# Gemma / GLM-style stop tokens used by this thinking model.
_STOP_TOKENS = [
    "<end_of_turn>",   # Gemma native EOS
    "<|eot_id|>",      # Llama 3 (kept as fallback)
    "<|end_of_text|>",
    "<|eom_id|>",
]

# Matches a full <think>...</think> block, including newlines.
# Also handles the GLM edge-case where the opening tag is injected by the
# chat template and omitted from the model output (only </think> is present).
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
    # Case 1: llama.cpp already separated the reasoning into its own field.
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""

    if reasoning:
        log.debug("[thinking] %s", reasoning[:500])
        return content.strip()

    # Case 2: thinking block is embedded in content.
    if "</think>" in content:
        # Capture everything up to and including the closing tag for logging.
        think_match = _THINK_RE.match(content.lstrip())
        if think_match:
            log.debug("[thinking] %s", think_match.group(0)[:500])

        clean = _THINK_RE.sub("", content).strip()
        return clean

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
            # Increased from 1024: thinking models consume a portion of this
            # budget on reasoning tokens before generating the actual reply.
            # 2048 ensures the response isn't cut off after a long think block.
            "max_tokens": 2048,
            "stop": _STOP_TOKENS,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        session = self._get_session()
        resp = await session.post(
            f"{self._base}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=cfg.model_timeout_seconds),
        )

        if resp.status == 500 and tools:
            log.warning("llama-server returned 500 with tools — retrying without tools")
            payload.pop("tools", None)
            payload.pop("tool_choice", None)
            resp = await session.post(
                f"{self._base}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=cfg.model_timeout_seconds),
            )

        resp.raise_for_status()
        data: dict[str, Any] = await resp.json()

        # Mutate the response in-place so all callers (agent._run_loop, etc.)
        # transparently receive clean content without thinking blocks.
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
