"""Async Groq API client — fast cloud inference fallback.

Used as an optional speed tier when GROQ_API_KEY is set.  Only non-sensitive
query types are routed here (see intent.is_groq_eligible).  Filesystem
operations, scheduler jobs, and diagnostic queries are always handled locally.

Groq's LPU delivers sub-100 ms TTFT and ~300-600 tok/s, roughly 40-80× faster
than CPU-only inference on the i5-10400H for eligible queries.

Free tier (as of 2025): 30 req/min, 6 000 req/day on llama-3.1-8b-instant.
Dashboard: https://console.groq.com/keys
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

import aiohttp

log = logging.getLogger(__name__)

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# llama-3.1-8b-instant: best TTFT on Groq free tier; strong enough for
# general chat and search synthesis.
_DEFAULT_MODEL = "llama-3.1-8b-instant"

TokenCallback = Callable[[str], Awaitable[None]]


class GroqClient:
    """Thin async wrapper around Groq's OpenAI-compatible chat completions API.

    Instantiated once in Agent when GROQ_API_KEY is configured; reused across
    requests.  The session is closed via close() on bot shutdown.

    The aiohttp session is created lazily on first use rather than in
    __init__.  Agent.__init__ runs inside LocalBot.__init__, which executes
    in main() *before* bot.run() starts the asyncio event loop — so
    aiohttp.ClientSession() in __init__ would raise "no running event loop".
    This mirrors the lazy _get_session() pattern already used in
    tools/search.py and tools/reddit.py.
    """

    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL) -> None:
        self._api_key = api_key
        self._model = model
        self._session: aiohttp.ClientSession | None = None

    @property
    def model(self) -> str:
        return self._model

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.3,
        on_token: TokenCallback | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant reply text.

        When *on_token* is provided the request is streamed and each content
        delta is forwarded to the callback as it arrives, matching the
        interface used by LlamaCppClient.

        Raises RuntimeError on non-2xx responses so the caller can fall back
        to the local model cleanly.
        """
        use_stream = on_token is not None
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": use_stream,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        session = self._get_session()
        try:
            resp = await session.post(
                _GROQ_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            )
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"Groq request failed: {exc}") from exc

        if resp.status >= 400:
            body = await resp.text()
            raise RuntimeError(f"Groq API returned HTTP {resp.status}: {body[:300]}")

        if use_stream:
            import json as _json

            full_content = ""
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = _json.loads(data_str)
                except _json.JSONDecodeError:
                    continue
                token = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                if token:
                    full_content += token
                    await on_token(token)
            return full_content.strip()

        data = await resp.json()
        return (data["choices"][0]["message"].get("content") or "").strip()
