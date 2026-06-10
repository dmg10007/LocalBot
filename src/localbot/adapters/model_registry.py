"""Hot-swap model registry.

Manages named model slots (general, coding, reasoning). Because the host
machine can only run one ~7-9B model at a time, only a single
llama-server process is live at any moment.  Switching slots stops the
current server, starts the new one, then restarts an idle timer that
reloads the general (lightweight) model after IDLE_UNLOAD_SECONDS of
inactivity.

Slot configuration is read from environment variables:

  SLOT_GENERAL_MODEL=models/llama-3.2-3b-instruct.gguf
  SLOT_GENERAL_PORT=8080
  SLOT_CODING_MODEL=models/qwen2.5-coder-7b-instruct.gguf
  SLOT_CODING_PORT=8081
  SLOT_REASONING_MODEL=models/deepseek-r1-8b.gguf
  SLOT_REASONING_PORT=8082

A slot whose MODEL path is empty is disabled; requests that would
route to it fall back to the general slot automatically.

All public methods are coroutine-safe: a single asyncio.Lock serialises
slot switches so that concurrent requests queue cleanly.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from localbot.adapters.llamacpp_client import LlamaCppClient
from localbot.adapters.llamacpp_server import LlamaCppServer
from localbot.config import cfg

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


@dataclass
class SlotConfig:
    name: str
    model_path: str
    port: int
    n_gpu_layers: int
    ctx_size: int
    threads: int
    extra_args: str

    @property
    def enabled(self) -> bool:
        return bool(self.model_path)


class ModelRegistry:
    """Lifecycle manager for named model slots with hot-swap support."""

    def __init__(self) -> None:
        self._slots: dict[str, SlotConfig] = self._build_slots()
        self._active_slot: str = "general"
        self._server: LlamaCppServer | None = None
        self._client: LlamaCppClient | None = None
        self._lock = asyncio.Lock()
        self._idle_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def active_slot(self) -> str:
        return self._active_slot

    @property
    def client(self) -> LlamaCppClient:
        """The client for the currently active slot. Always valid after ensure_slot()."""
        if self._client is None:
            raise RuntimeError("ModelRegistry: no client — call ensure_slot() first")
        return self._client

    async def ensure_slot(self, slot: str) -> LlamaCppClient:
        """Guarantee that *slot* is the active running server.

        If a different slot is active it is stopped first (hot-swap).
        Returns the ready LlamaCppClient for the new slot.
        Concurrent callers queue behind the internal lock.
        """
        resolved = self._resolve(slot)
        async with self._lock:
            if self._active_slot == resolved and self._client is not None and self._client.is_ready:
                self._reset_idle_timer()
                return self._client

            await self._stop_active()
            await self._start_slot(resolved)
            self._reset_idle_timer()
            return self._client  # type: ignore[return-value]

    async def start_general(self) -> None:
        """Start the general slot on initial bot startup."""
        await self.ensure_slot("general")

    async def stop_all(self) -> None:
        """Gracefully stop whichever server is currently running."""
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
        async with self._lock:
            await self._stop_active()

    def slot_enabled(self, slot: str) -> bool:
        return self._slots.get(slot, SlotConfig("", "", 0, 0, 0, 0, "")).enabled

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, slot: str) -> str:
        """Return *slot* if enabled, otherwise fall back to 'general'."""
        if slot in self._slots and self._slots[slot].enabled:
            return slot
        if slot != "general":
            log.info("Slot '%s' not configured — falling back to 'general'", slot)
        return "general"

    async def _stop_active(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
        if self._server is not None:
            await self._server.stop()
            self._server = None

    async def _start_slot(self, slot: str) -> None:
        sc = self._slots[slot]
        log.info("[ModelRegistry] Loading slot '%s' (%s)", slot, sc.model_path)
        server = LlamaCppServer(
            model_path=sc.model_path,
            port=sc.port,
            n_gpu_layers=sc.n_gpu_layers,
            ctx_size=sc.ctx_size,
            threads=sc.threads,
            extra_args=sc.extra_args,
        )
        client = LlamaCppClient(host=cfg.llama_server_host, port=sc.port)
        await server.start()
        await client.wait_until_ready(retries=20, delay=1.5)
        self._server = server
        self._client = client
        self._active_slot = slot
        log.info("[ModelRegistry] Slot '%s' is ready", slot)

    def _reset_idle_timer(self) -> None:
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
        self._idle_task = asyncio.create_task(self._idle_unload())
        self._idle_task.add_done_callback(self._on_idle_task_done)

    async def _idle_unload(self) -> None:
        await asyncio.sleep(cfg.idle_unload_seconds)
        if self._active_slot == "general":
            return
        log.info(
            "[ModelRegistry] Idle timeout (%ds) — swapping back to 'general'",
            cfg.idle_unload_seconds,
        )
        async with self._lock:
            await self._stop_active()
            await self._start_slot("general")

    def _on_idle_task_done(self, task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is not None:
            log.exception(
                "[ModelRegistry] Idle unload task failed",
                exc_info=task.exception(),
            )

    # ------------------------------------------------------------------
    # Slot configuration builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_slots() -> dict[str, SlotConfig]:
        """Read SLOT_<NAME>_* env vars and return a slot config dict.

        The general slot always exists and falls back to the legacy
        LLAMA_SERVER_* variables so that existing .env files continue
        to work without modification.
        """
        return {
            "general": SlotConfig(
                name="general",
                model_path=cfg.slot_general_model,
                port=cfg.slot_general_port,
                n_gpu_layers=cfg.llama_server_n_gpu_layers,
                ctx_size=cfg.llama_server_ctx_size,
                threads=cfg.llama_server_threads,
                extra_args=cfg.llama_server_extra_args,
            ),
            "coding": SlotConfig(
                name="coding",
                model_path=cfg.slot_coding_model,
                port=cfg.slot_coding_port,
                n_gpu_layers=cfg.llama_server_n_gpu_layers,
                ctx_size=cfg.llama_server_ctx_size,
                threads=cfg.llama_server_threads,
                extra_args=cfg.llama_server_extra_args,
            ),
            "reasoning": SlotConfig(
                name="reasoning",
                model_path=cfg.slot_reasoning_model,
                port=cfg.slot_reasoning_port,
                n_gpu_layers=cfg.llama_server_n_gpu_layers,
                ctx_size=cfg.llama_server_ctx_size,
                threads=cfg.llama_server_threads,
                extra_args=cfg.llama_server_extra_args,
            ),
        }
