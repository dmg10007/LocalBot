"""Async HTTP client for the llama-server OpenAI-compatible API.

Auto-detects the loaded model family on startup and applies the correct
stop tokens and think-stripping strategy for that family. Swapping the
model in .env is all that is needed — no code changes required.
"""
from __future__ import annotations

import asyncio
import logging
import re
from enum import Enum, auto
from typing import Any

import aiohttp

from localbot.config import cfg

log = logging.getLogger(__name__)


class ModelFamily(Enum):
    GEMMA = auto()      # Google Gemma / Gemma-based thinking models
    LLAMA = auto()      # Meta Llama 2 / 3 / 3.1 / 3.2
    MISTRAL = auto()    # Mistral / Mixtral
    QWEN = auto()       # Alibaba Qwen 1 / 2 / 2.5 (also emits <think>)
    DEEPSEEK = auto()   # DeepSeek / DeepSeek-R1 (thinking model)
    PHI = auto()        # Microsoft Phi-2 / Phi-3 / Phi-3.5
    UNKNOWN = auto()    # Unrecognised — safe defaults applied


# Name-pattern → family. Checked in order; first match wins.
_FAMILY_PATTERNS: list[tuple[re.Pattern[str], ModelFamily]] = [
    (re.compile(r"gemma|glm",        re.I), ModelFamily.GEMMA),
    (re.compile(r"llama",            re.I), ModelFamily.LLAMA),
    (re.compile(r"mistral|mixtral",  re.I), ModelFamily.MISTRAL),
    (re.compile(r"qwen",             re.I), ModelFamily.QWEN),
    (re.compile(r"deepseek",         re.I), ModelFamily.DEEPSEEK),
    (re.compile(r"phi",              re.I), ModelFamily.PHI),
]

# Stop tokens per family.
_STOP_TOKENS: dict[ModelFamily, list[str]] = {
    ModelFamily.GEMMA:    ["<end_of_turn>", "<eos>"],
    ModelFamily.LLAMA:    ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"],
    ModelFamily.MISTRAL:  ["</s>", "[INST]"],
    ModelFamily.QWEN:     ["<|im_end|>", "<|endoftext|>"],
    ModelFamily.DEEPSEEK: ["<└┘>", "<|end_of_sentence|>"],
    ModelFamily.PHI:      ["<|end|>", "<|endoftext|>"],
    ModelFamily.UNKNOWN:  [],
}

_THINKING_FAMILIES = {ModelFamily.GEMMA, ModelFamily.DEEPSEEK, ModelFamily.QWEN}

_THINK_RE = re.compile(
    r"(?:<think>)?.*?</think>",
    re.DOTALL | re.IGNORECASE,
)


def _detect_family_from_name(model_name: str) -> ModelFamily:
    for pattern, family in _FAMILY_PATTERNS:
        if pattern.search(model_name):
            return family
    return ModelFamily.UNKNOWN


def strip_thinking(message: dict[str, Any]) -> str:
    """Extract the user-facing reply, discarding any <think> reasoning block."""
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
        self._family: ModelFamily = ModelFamily.UNKNOWN
        self._model_name: str = "unknown"

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def detect_model(self) -> None:
        """Query /v1/models and classify the loaded model family.

        Called once after the server is healthy. An env override
        (LLAMA_SERVER_MODEL_FAMILY) takes priority over auto-detection
        for fine-tunes or models with unusual filenames.
        """
        override = cfg.llama_server_model_family.lower().strip()
        if override:
            family_map = {
                "gemma":    ModelFamily.GEMMA,
                "llama":    ModelFamily.LLAMA,
                "mistral":  ModelFamily.MISTRAL,
                "qwen":     ModelFamily.QWEN,
                "deepseek": ModelFamily.DEEPSEEK,
                "phi":      ModelFamily.PHI,
            }
            if override in family_map:
                self._family = family_map[override]
                self._model_name = f"(override: {override})"
                log.info("Model family overridden via env: %s", self._family.name)
                return
            log.warning("Unknown LLAMA_SERVER_MODEL_FAMILY value '%s' — falling back to auto-detect.", override)

        session = self._get_session()
        try:
            async with session.get(
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
            "Detected model: '%s' → family=%s (stop=%s, think_strip=%s)",
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
        """Call /v1/chat/completions with family-appropriate settings."""
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

        session = self._get_session()
        resp = await session.post(
            f"{self._base}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=cfg.model_timeout_seconds),
        )
        resp.raise_for_status()
        data: dict[str, Any] = await resp.json()

        is_thinking_model = self._family in _THINKING_FAMILIES
        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            if not msg.get("tool_calls"):
                if is_thinking_model:
                    msg["content"] = strip_thinking(msg)
                else:
                    msg["content"] = (msg.get("content") or "").strip()

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
                        # Fix #13: detect_model only needs to run once — skip
                        # subsequent calls once the family has been identified.
                        if self._family is ModelFamily.UNKNOWN:
                            await self.detect_model()
                        return
            except Exception:
                pass
            log.debug("Waiting for llama-server... (%d/%d)", attempt + 1, retries)
            await asyncio.sleep(delay)
        raise RuntimeError("llama-server did not become ready in time")
