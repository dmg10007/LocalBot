"""Hot-swap model registry.

Manages a single active llama-server slot.  At most one slot runs at a
time — swapping kills the current process, starts the new one, then
resets the idle-unload timer.

Slot names
----------
``general``   — lightweight always-on fallback (Llama 3.2 3B, etc.)
``coding``    — code-optimised model (Qwen2.5-Coder, etc.)
``reasoning`` — chain-of-thought model (DeepSeek-R1, etc.)

A slot whose model path is empty/None is disabled; requests that would
route there fall back to ``general``.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

from localbot.adapters.llamacpp_client import LlamaCppClient
from localbot.adapters.llamacpp_server import LlamaCppServer
from localbot.config import cfg

log = logging.getLogger(__name__)

SlotName = Literal["general", "coding", "reasoning"]

# How long to wait for llama-server to become healthy after launch.
# 120 retries × 1 s = 2 minutes — enough for an 8B Q4 model on CPU.
_READY_RETRIES = 120
_READY_DELAY = 1.0


@dataclass
class _SlotConfig:
    name: SlotName
    model_path: str
    port: int


class ModelRegistry:
    """Manages a single active llama-server slot with hot-swap support.

    Usage::

        registry = ModelRegistry()
        client = await registry.acquire("coding")
        reply = await client.chat(messages)
        # idle timer will unload after cfg.idle_unload_seconds of inactivity
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._active_slot: SlotName | None = None
        self._server: LlamaCppServer | None = None
        self._client: LlamaCppClient | None = None
        self._idle_task: asyncio.Task | None = None

        # Build slot configs from environment, falling back to legacy vars.
        self._slots: dict[SlotName, _SlotConfig] = {
            "general": _SlotConfig(
                name="general",
                model_path=cfg.slot_general_model,
                port=cfg.slot_general_port,
            ),
            "coding": _SlotConfig(
                name="coding",
                model_path=cfg.slot_coding_model,
                port=cfg.slot_coding_port,
            ),
            "reasoning": _SlotConfig(
                name="reasoning",
                model_path=cfg.slot_reasoning_model,
                port=cfg.slot_reasoning_port,
            ),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_slot_available(self, slot: SlotName) -> bool:
        """Return True if the slot has a model path configured."""
        return bool(self._slots[slot].model_path)

    async def acquire(self, slot: SlotName) -> LlamaCppClient:
        """Return a ready LlamaCppClient for *slot*, swapping if necessary.

        Falls back to ``general`` if *slot* is not configured.  Callers
        that hold the returned client should not cache it across turns —
        re-acquire each time so the registry can swap freely.
        """
        if not self.is_slot_available(slot):
            log.debug("Slot '%s' not configured — falling back to general", slot)
            slot = "general"

        async with self._lock:
            if self._active_slot != slot:
                await self._swap_to(slot)
            self._reset_idle_timer()
            assert self._client is not None
            return self._client

    async def warm_general(self) -> None:
        """Start the general slot at boot so the bot is immediately ready."""
        await self.acquire("general")

    async def shutdown(self) -> None:
        """Stop the active server and cancel the idle timer."""
        self._cancel_idle_timer()
        async with self._lock:
            await self._stop_current()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _swap_to(self, slot: SlotName) -> None:
        """Stop the current slot and start *slot*.  Must be called under lock."""
        if self._active_slot is not None:
            log.info(
                "[registry] swapping %s → %s", self._active_slot, slot
            )
        await self._stop_current()

        sc = self._slots[slot]
        server = LlamaCppServer(
            model_path=sc.model_path,
            port=sc.port,
        )
        # Use cfg.llama_server_client_host (default 127.0.0.1) — NOT
        # cfg.llama_server_host which is the bind address (0.0.0.0) and
        # is not a routable destination.
        client = LlamaCppClient(
            host=cfg.llama_server_client_host,
            port=sc.port,
        )

        log.info("[registry] starting slot '%s' (port %d)", slot, sc.port)
        await server.start()
        await client.wait_until_ready(retries=_READY_RETRIES, delay=_READY_DELAY)

        self._server = server
        self._client = client
        self._active_slot = slot
        log.info("[registry] slot '%s' ready", slot)

    async def _stop_current(self) -> None:
        """Gracefully stop the current server/client.  Must be called under lock."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                log.exception("[registry] error closing client for slot '%s'", self._active_slot)
            self._client = None
        if self._server is not None:
            try:
                await self._server.stop()
            except Exception:
                log.exception("[registry] error stopping server for slot '%s'", self._active_slot)
            self._server = None
        self._active_slot = None

    # ------------------------------------------------------------------
    # Idle-unload timer
    # ------------------------------------------------------------------

    def _reset_idle_timer(self) -> None:
        self._cancel_idle_timer()
        if cfg.idle_unload_seconds > 0:
            self._idle_task = asyncio.create_task(self._idle_worker())
            self._idle_task.add_done_callback(self._on_idle_task_done)

    def _cancel_idle_timer(self) -> None:
        if self._idle_task is not None and not self._idle_task.done():
            self._idle_task.cancel()
        self._idle_task = None

    async def _idle_worker(self) -> None:
        await asyncio.sleep(cfg.idle_unload_seconds)
        async with self._lock:
            if self._active_slot in ("coding", "reasoning"):
                log.info(
                    "[registry] idle timeout — unloading '%s', reloading 'general'",
                    self._active_slot,
                )
                await self._swap_to("general")

    def _on_idle_task_done(self, task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is not None:
            log.exception(
                "[registry] idle worker raised unexpectedly",
                exc_info=task.exception(),
            )
