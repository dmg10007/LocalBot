"""Discord bot entry-point and on_message handler.

Refactored changes
------------------
* Command dispatch extracted to commands.py — app.py no longer defines
  individual command functions.
* `registry` and `scheduler` are public properties so commands.py can
  access them without underscore attribute hacks.
* Rate-limit table is evicted lazily via a time-based cutoff (unchanged
  logic, but now isolated in _RateLimiter to make it testable).
* Backend startup failures surface via log.exception instead of bare
  log.exception inside a hidden callback chain.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import discord

from localbot.adapters.llamacpp_downloader import download_and_install
from localbot.adapters.llamacpp_updater import check_for_update
from localbot.adapters.model_registry import ModelRegistry
from localbot.agent import Agent
from localbot.commands import COMMAND_HANDLERS
from localbot.config import cfg
from localbot.messaging import split_message
from localbot.scheduler.service import SchedulerService
from localbot.storage.db import init_db
from localbot.tools import search as search_module, reddit as reddit_module

log = logging.getLogger(__name__)


class _RateLimiter:
    """Per-user cooldown tracker backed by a plain dict.

    Thread-safety note: on_message always runs on the asyncio event loop
    so no lock is needed here.
    """

    def __init__(self, window: float) -> None:
        self._window = window
        self._last: dict[str, float] = {}

    def is_limited(self, user_id: str) -> float:
        """Return remaining cooldown seconds, or 0 if the user is allowed."""
        now = time.monotonic()
        self._evict(now)
        remaining = self._window - (now - self._last.get(user_id, 0.0))
        return max(0.0, remaining)

    def record(self, user_id: str) -> None:
        self._last[user_id] = time.monotonic()

    def _evict(self, now: float) -> None:
        cutoff = now - self._window * 10
        stale = [uid for uid, ts in self._last.items() if ts < cutoff]
        for uid in stale:
            del self._last[uid]


class LocalBot(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self._registry = ModelRegistry()
        self._scheduler = SchedulerService(self._send_scheduled)
        self._agent = Agent(self._registry, scheduler=self._scheduler)
        self._rate_limiter = _RateLimiter(cfg.rate_limit_seconds)
        self._backend_ready = False
        self._backend_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Public accessors (used by commands.py)
    # ------------------------------------------------------------------

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    @property
    def scheduler(self) -> SchedulerService:
        return self._scheduler

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_ready(self) -> None:
        log.info("Logged in as %s (id=%s)", self.user, self.user.id if self.user else "?")
        self._backend_task = asyncio.create_task(self._start_backend())
        self._backend_task.add_done_callback(self._on_backend_done)

    def _on_backend_done(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled() and (exc := task.exception()) is not None:
            log.exception("Backend startup task failed", exc_info=exc)

    async def _start_backend(self) -> None:
        try:
            await _check_for_llama_update()
            await self._registry.warm_general()
            self._scheduler.start()
            self._backend_ready = True
            log.info("Backend ready")
        except Exception:
            log.exception("Backend startup failed")

    async def close(self) -> None:
        self._scheduler.stop()
        await search_module.close_session()
        await reddit_module.close_session()
        await self._registry.shutdown()
        if self._agent._groq is not None:
            await self._agent._groq.close()
        await super().close()

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        if not isinstance(message.channel, discord.DMChannel):
            return

        user_id = str(message.author.id)
        text = message.content.strip()

        if await self._dispatch_command(message, user_id, text):
            return

        if not self._backend_ready:
            await message.channel.send("Still starting up — please try again in a moment.")
            return

        if len(text) > cfg.max_input_length:
            await message.channel.send(
                f"Your message is too long (max {cfg.max_input_length} characters). Please shorten it."
            )
            return

        remaining = self._rate_limiter.is_limited(user_id)
        if remaining > 0:
            await message.channel.send(
                f"Please wait {remaining:.1f}s before sending another message."
            )
            return
        self._rate_limiter.record(user_id)

        async with message.channel.typing():
            reply = await self._agent.handle(user_id, text)

        for chunk in split_message(reply):
            await message.channel.send(chunk)

    async def _dispatch_command(
        self, message: discord.Message, user_id: str, text: str
    ) -> bool:
        for handler in COMMAND_HANDLERS:
            if await handler(self, message, user_id, text):
                return True
        return False

    async def _send_scheduled(self, user_id: str, prompt: str) -> None:
        try:
            discord_user = await self.fetch_user(int(user_id))
            dm = await discord_user.create_dm()
            async with dm.typing():
                reply = await self._agent.handle(user_id, prompt)
            for chunk in split_message(reply):
                await dm.send(chunk)
        except Exception:
            log.exception("Failed to deliver scheduled message to user %s", user_id)


# ---------------------------------------------------------------------------
# Auto-updater helpers
# ---------------------------------------------------------------------------

def _prompt_stdin(prompt: str) -> str:
    return input(prompt)


async def _ask_terminal(prompt: str, timeout: float) -> str | None:
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _prompt_stdin, prompt),
            timeout=timeout,
        )
    except (asyncio.TimeoutError, EOFError):
        return None


async def _check_for_llama_update(*, _already_updated: bool = False) -> None:
    if not cfg.llama_update_check:
        log.debug("llama.cpp update check disabled")
        return

    log.info("Checking for llama.cpp updates…")
    info = await check_for_update(
        cfg.llama_server_executable,
        timeout_seconds=float(cfg.llama_update_check_timeout_seconds),
    )
    if info is None:
        log.warning("llama.cpp update check failed (network unavailable or rate-limited)")
        return

    current_str = f"b{info.current}" if info.current is not None else "unknown"
    if not info.available:
        log.info("llama.cpp is up to date (%s, latest b%s)", current_str, info.latest)
        return

    log.warning("llama.cpp update available: %s → b%s (%s)", current_str, info.latest, info.url)
    if _already_updated:
        return

    if cfg.llama_update_auto:
        do_update = True
        log.info("LLAMA_UPDATE_AUTO=true — installing automatically.")
    else:
        answer = await _ask_terminal(
            f"\nInstall llama.cpp b{info.latest}? [y/N] "
            f"(auto-skip in {cfg.llama_update_prompt_timeout_seconds}s): ",
            timeout=float(cfg.llama_update_prompt_timeout_seconds),
        )
        if answer is None:
            log.info("Update prompt timed out — continuing without update.")
            return
        do_update = answer.strip().lower() in ("y", "yes")

    if not do_update:
        log.info("Update declined.")
        return

    install_dir = Path(cfg.llama_server_executable).parent
    log.info("Installing llama.cpp b%s into %s…", info.latest, install_dir)
    result = await download_and_install(install_dir, timeout_seconds=120.0)
    if result.ok:
        log.info("llama.cpp updated: %s", result.message)
        await _check_for_llama_update(_already_updated=True)
    else:
        log.warning("llama.cpp update failed: %s", result.message)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    init_db()
    bot = LocalBot()
    bot.run(cfg.discord_bot_token)
