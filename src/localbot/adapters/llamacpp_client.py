"""Async HTTP client for the llama-server OpenAI-compatible API."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from localbot.config import cfg

log = logging.getLogger(__name__)


class LlamaCppClient:
    def __init__(self) -> None:
        self._base = f"http://{cfg.llama_server_host}:{cfg.llama_server_port}"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Call /v1/chat/completions. Falls back to no-tools on 500."""
        payload: dict[str, Any] = {
            "messages": messages,
            "stream": False,
            "temperature": 0.7,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with aiohttp.ClientSession() as session:
            resp = await session.post(
                f"{self._base}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=cfg.model_timeout_seconds),
            )

            # Some models return 500 when given tool schemas they don't support.
            # Retry once without tools so the user still gets a reply.
            if resp.status == 500 and tools:
                log.warning(
                    "llama-server returned 500 with tools — retrying without tool_choice"
                )
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
        """Poll /health until llama-server is accepting requests."""
        async with aiohttp.ClientSession() as session:
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
