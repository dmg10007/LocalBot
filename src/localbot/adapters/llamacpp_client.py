"""Async HTTP client for the llama-server OpenAI-compatible API.

A single shared aiohttp.ClientSession is created at first use and reused
for all subsequent requests, avoiding per-call TCP overhead.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from localbot.config import cfg

log = logging.getLogger(__name__)

# Llama 3 end-of-turn / end-of-sequence stop tokens.
# Without these, the model can continue generating past its natural stop point
# after receiving tool results, producing garbled or repeated output.
_LLAMA3_STOP = ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"]


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
        """Call /v1/chat/completions. Falls back to no-tools on 500."""
        payload: dict[str, Any] = {
            "messages": messages,
            "stream": False,
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 1024,
            "stop": _LLAMA3_STOP,
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
        return await resp.json()  # type: ignore[no-any-return]

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
