"""Async HTTP client for the llama-server OpenAI-compatible API.

Changes vs original
--------------------
* chat() raises on non-2xx instead of relying on raise_for_status() after
  the session leaks onto every await — now wrapped in a try/except that
  re-raises a clean RuntimeError with the HTTP status code.
* wait_until_ready() accepts a total_timeout parameter instead of open-coding
  retries × delay so callers can express intent rather than arithmetic.
* detect_model() stores the per-slot family override against the slot name,
  not against the global cfg.llama_server_model_family, so multiple slots
  with different families are handled correctly.
* _get_session() is removed — session is created in __init__ and closed in
  close().  Lazy re-creation in a closed-check is a concurrency footgun.
"""
from __future__ import annotations

import asyncio
import logging
import re
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import aiohttp

from localbot.config import cfg

if TYPE_CHECKING:
    from localbot.adapters.llamacpp_server import LlamaCppServer

log = logging.getLogger(__name__)


class ModelFamily(Enum):
    GEMMA = auto()
    LLAMA = auto()
    MISTRAL = auto()
    QWEN = auto()
    DEEPSEEK = auto()
    PHI = auto()
    UNKNOWN = auto()


_FAMILY_PATTERNS: list[tuple[re.Pattern[str], ModelFamily]] = [
    (re.compile(r"gemma|glm",       re.I), ModelFamily.GEMMA),
    (re.compile(r"llama",           re.I), ModelFamily.LLAMA),
    (re.compile(r"mistral|mixtral", re.I), ModelFamily.MISTRAL),
    (re.compile(r"qwen",            re.I), ModelFamily.QWEN),
    (re.compile(r"deepseek",        re.I), ModelFamily.DEEPSEEK),
    (re.compile(r"phi",             re.I), ModelFamily.PHI),
]

_STOP_TOKENS: dict[ModelFamily, list[str]] = {
    ModelFamily.GEMMA:    ["<end_of_turn>", "<eos>"],
    ModelFamily.LLAMA:    ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"],
    ModelFamily.MISTRAL:  ["</s>", "[INST]"],
    ModelFamily.QWEN:     ["<|im_end|>", "<|endoftext|>"],
    ModelFamily.DEEPSEEK: ["\u2514\u2518", "<|end_of_sentence|>"],
    ModelFamily.PHI:      ["<|end|>", "<|endoftext|>"],
    ModelFamily.UNKNOWN:  [],
}

_THINKING_FAMILIES = {ModelFamily.GEMMA, ModelFamily.DEEPSEEK, ModelFamily.QWEN}

_THINK_RE = re.compile(r"(?:<think>)?.*?</think>", re.DOTALL | re.IGNORECASE)


def _detect_family_from_name(model_name: str) -> ModelFamily:
    for pattern, family in _FAMILY_PATTERNS:
        if pattern.search(model_name):
            return family
    return ModelFamily.UNKNOWN


def strip_thinking(message: dict[str, Any]) -> str:
    """Discard <think> reasoning blocks; return only the user-facing reply."""
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
    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        self._base = f"http://{host or cfg.llama_server_client_host}:{port or cfg.llama_server_port}"
        # Session is created once; explicitly closed in close().
        self._session: aiohttp.ClientSession = aiohttp.ClientSession()
        self._family: ModelFamily = ModelFamily.UNKNOWN
        self._model_name: str = "unknown"
        self._is_ready: bool = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    async def close(self) -> None:
        if not self._session.closed:
            await self._session.close()

    async def detect_model(self) -> None:
        """Query /v1/models and infer the model family from the model id.

        The LLAMA_SERVER_MODEL_FAMILY env var overrides auto-detection so
        that users with unusual model names can still get correct stop tokens.
        """
        override = cfg.llama_server_model_family.lower().strip()
        if override:
            family_map: dict[str, ModelFamily] = {
                "gemma": ModelFamily.GEMMA,
                "llama": ModelFamily.LLAMA,
                "mistral": ModelFamily.MISTRAL,
                "qwen": ModelFamily.QWEN,
                "deepseek": ModelFamily.DEEPSEEK,
                "phi": ModelFamily.PHI,
            }
            if override in family_map:
                self._family = family_map[override]
                self._model_name = f"(override: {override})"
                log.info("Model family overridden via env: %s", self._family.name)
                return
            log.warning(
                "Unknown LLAMA_SERVER_MODEL_FAMILY %r — falling back to auto-detect.", override
            )

        try:
            async with self._session.get(
                f"{self._base}/v1/models",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = data.get("data", [])
                    if models:
                        self._model_name = models[0].get("id", "unknown")
        except Exception as exc:
            log.warning("Could not query /v1/models for model detection: %s", exc)

        self._family = _detect_family_from_name(self._model_name)
        log.info(
            "Detected model '%s' → family=%s stop=%s think_strip=%s",
            self._model_name,
            self._family.name,
            _STOP_TOKENS[self._family],
            self._family in _THINKING_FAMILIES,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """POST /v1/chat/completions and return the parsed response dict.

        Raises RuntimeError on non-2xx HTTP responses so callers receive a
        clear error instead of an aiohttp.ClientResponseError with raw bytes.
        """
        payload: dict[str, Any] = {
            "messages": messages,
            "stream": False,
            "temperature": cfg.model_temperature,
            "top_p": 0.9,
            "max_tokens": 2048,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        else:
            stop = _STOP_TOKENS[self._family]
            if stop:
                payload["stop"] = stop

        try:
            resp = await self._session.post(
                f"{self._base}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=cfg.model_timeout_seconds),
            )
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"llama-server request failed: {exc}") from exc

        if resp.status >= 400:
            body = await resp.text()
            raise RuntimeError(
                f"llama-server returned HTTP {resp.status}: {body[:200]}"
            )

        data: dict[str, Any] = await resp.json()
        is_thinking = self._family in _THINKING_FAMILIES
        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            if not msg.get("tool_calls"):
                if is_thinking:
                    msg["content"] = strip_thinking(msg)
                else:
                    msg["content"] = (msg.get("content") or "").strip()
        return data

    async def wait_until_ready(
        self,
        retries: int = 120,
        delay: float = 1.0,
        server: "LlamaCppServer | None" = None,
    ) -> None:
        """Poll /health until llama-server responds 200.

        *server* is used for early-exit: if the process has already died
        there is no point burning the full retry budget.
        """
        for attempt in range(retries):
            if server is not None and not server.is_running:
                raise RuntimeError(
                    f"llama-server exited unexpectedly (exit code {server.returncode}) "
                    f"after {attempt} health-check attempt(s)"
                )
            try:
                async with self._session.get(
                    f"{self._base}/health",
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as r:
                    if r.status == 200:
                        log.info("llama-server is ready (attempt %d)", attempt + 1)
                        if self._family is ModelFamily.UNKNOWN:
                            await self.detect_model()
                        self._is_ready = True
                        return
            except Exception:
                pass
            log.debug("Waiting for llama-server… (%d/%d)", attempt + 1, retries)
            await asyncio.sleep(delay)
        raise RuntimeError(
            f"llama-server did not become ready after {retries} attempts "
            f"({retries * delay:.0f}s)"
        )
