"""Entry point, Discord client, and command handler."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Callable, Awaitable

import discord

from localbot.adapters.llamacpp_downloader import download_and_install
from localbot.adapters.llamacpp_updater import check_for_update
from localbot.adapters.model_registry import ModelRegistry
from localbot.agent import Agent
from localbot.config import cfg
from localbot.scheduler.service import SchedulerService
from localbot.scheduler.store import get_user_timezone, set_user_timezone
from localbot.storage.db import init_db
from localbot.storage.history import clear_history
from localbot.tools import search as search_module, reddit as reddit_module
from localbot.tools.time_tools import get_current_time
from localbot.messaging import split_message

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Command handler type
# ---------------------------------------------------------------------------

_CommandHandler = Callable[["LocalBot", discord.Message, str, str], Awaitable[bool]]


async def _cmd_jobs_list(bot: "LocalBot", message: discord.Message, user_id: str, text: str) -> bool:
    if text.lower() != "jobs list":
        return False
    jobs = bot._scheduler.list_user_jobs(user_id)
    if not jobs:
        await message.channel.send("You have no scheduled jobs.")
    else:
        lines = [f"`{j.job_id}` — `{j.cron_expr}` — {j.prompt}" for j in jobs]
        await message.channel.send("**Your scheduled jobs:**\n" + "\n".join(lines))
    return True


async def _cmd_jobs_cancel(bot: "LocalBot", message: discord.Message, user_id: str, text: str) -> bool:
    m = re.match(r"^jobs cancel ([a-zA-Z0-9_-]+)$", text, re.IGNORECASE)
    if not m:
        return False
    cancelled = bot._scheduler.cancel_job(m.group(1))
    await message.channel.send(
        f"Job `{m.group(1)}` cancelled." if cancelled else "Job not found."
    )
    return True


async def _cmd_timezone_set(bot: "LocalBot", message: discord.Message, user_id: str, text: str) -> bool:
    m = re.match(r"^timezone set (.+)$", text, re.IGNORECASE)
    if not m:
        return False
    tz = m.group(1).strip()
    try:
        set_user_timezone(user_id, tz)
    except ValueError:
        await message.channel.send(
            f"Unknown timezone `{tz}`. Use an IANA name like `America/New_York`."
        )
        return True
    await message.channel.send(f"Timezone set to `{tz}`.")
    return True


async def _cmd_timezone_show(bot: "LocalBot", message: discord.Message, user_id: str, text: str) -> bool:
    if text.lower() != "timezone show":
        return False
    tz = get_user_timezone(user_id)
    await message.channel.send(f"Your timezone is `{tz}`.")
    return True


async def _cmd_time_now(bot: "LocalBot", message: discord.Message, user_id: str, text: str) -> bool:
    if text.lower() != "time now":
        return False
    tz = get_user_timezone(user_id)
    await message.channel.send(get_current_time(tz))
    return True


async def _cmd_clear(bot: "LocalBot", message: discord.Message, user_id: str, text: str) -> bool:
    if text.lower() not in ("clear", "clear history", "/clear"):
        return False
    clear_history(user_id)
    await message.channel.send("Conversation history cleared.")
    return True


async def _cmd_model_status(bot: "LocalBot", message: discord.Message, user_id: str, text: str) -> bool:
    if text.lower() not in ("model status", "model"):
        return False
    registry = bot._registry
    active = registry._active_slot or "none"
    slots = []
    for name in ("general", "coding", "reasoning"):
        available = registry.is_slot_available(name)  # type: ignore[arg-type]
        status = "active" if name == active else ("available" if available else "not configured")
        slots.append(f"  `{name}` — {status}")
    await message.channel.send("**Model slots:**\n" + "\n".join(slots))
    return True


async def _cmd_help(bot: "LocalBot", message: discord.Message, user_id: str, text: str) -> bool:
    if text.lower() not in ("help", "/help"):
        return False
    await message.channel.send(
        "**LocalBot commands**\n"
        "`jobs list` — List your scheduled jobs\n"
        "`jobs cancel <id>` — Cancel a scheduled job\n"
        "`timezone set <IANA>` — Set your timezone (e.g. `America/New_York`)\n"
        "`timezone show` — Show your current timezone\n"
        "`time now` — Show current time in your timezone\n"
        "`model status` — Show active and configured model slots\n"
        "`clear` — Clear your conversation history\n"
        "\nFor scheduled reminders, just ask naturally:\n"
        "> *Remind me every morning at 8am to review my task list*\n"
        "\nFor coding tasks, just describe what you need:\n"
        "> *Fix the bug in src/app.py line 42*\n"
        "> *Commit the updated README to my repo and open a PR*"
    )
    return True


_COMMAND_HANDLERS: list[_CommandHandler] = [
    _cmd_jobs_list,
    _cmd_jobs_cancel,
    _cmd_timezone_set,
    _cmd_timezone_show,
    _cmd_time_now,
    _cmd_clear,
    _cmd_model_status,
    _cmd_help,
]


class LocalBot(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self._registry = ModelRegistry()
        self._scheduler = SchedulerService(self._send_scheduled)
        self._agent = Agent(self._registry, scheduler=self._scheduler)
        self._backend_ready = False
        self._last_request: dict[str, float] = {}
        self._backend_task: asyncio.Task | None = None

    async def on_ready(self) -> None:
        log.info("Logged in as %s (id=%s)", self.user, self.user.id if self.user else "?")
        self._backend_task = asyncio.create_task(self._start_backend())
        self._backend_task.add_done_callback(self._on_backend_task_done)

    def _on_backend_task_done(self, task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is not None:
            log.exception(
                "Backend startup task failed unexpectedly",
                exc_info=task.exception(),
            )

    async def _start_backend(self) -> None:
        try:
            await _check_for_llama_update()
            await self._registry.warm_general()
            self._scheduler.start()
            self._backend_ready = True
            log.info("Backend ready")
        except Exception:
            log.exception("Backend startup failed")

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        if not isinstance(message.channel, discord.DMChannel):
            return

        user_id = str(message.author.id)
        text = message.content.strip()

        if await self._handle_command(message, user_id, text):
            return

        if not self._backend_ready:
            await message.channel.send("Still starting up — please try again in a moment.")
            return

        if len(text) > cfg.max_input_length:
            await message.channel.send(
                f"Your message is too long (max {cfg.max_input_length} characters). Please shorten it."
            )
            return

        now = time.monotonic()
        self._evict_stale_rate_limit_entries(now)
        last = self._last_request.get(user_id, 0.0)
        if now - last < cfg.rate_limit_seconds:
            remaining = cfg.rate_limit_seconds - (now - last)
            await message.channel.send(
                f"Please wait {remaining:.1f}s before sending another message."
            )
            return
        self._last_request[user_id] = now

        async with message.channel.typing():
            reply = await self._agent.handle(user_id, text)

        for chunk in split_message(reply):
            await message.channel.send(chunk)

    def _evict_stale_rate_limit_entries(self, now: float) -> None:
        cutoff = now - cfg.rate_limit_seconds * 10
        stale = [uid for uid, ts in self._last_request.items() if ts < cutoff]
        for uid in stale:
            del self._last_request[uid]

    async def close(self) -> None:
        self._scheduler.stop()
        await search_module.close_session()
        await reddit_module.close_session()
        await self._registry.shutdown()
        await super().close()

    async def _handle_command(self, message: discord.Message, user_id: str, text: str) -> bool:
        for handler in _COMMAND_HANDLERS:
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
# Helpers
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
    except asyncio.TimeoutError:
        print()
        return None
    except EOFError:
        return None


async def _check_for_llama_update(*, _already_updated: bool = False) -> None:
    """Run the llama.cpp update check and optionally install the update."""
    if not cfg.llama_update_check:
        log.debug("llama.cpp update check disabled (LLAMA_UPDATE_CHECK=false)")
        return

    log.info("Checking for llama.cpp updates...")
    info = await check_for_update(
        cfg.llama_server_executable,
        timeout_seconds=float(cfg.llama_update_check_timeout_seconds),
    )

    if info is None:
        log.warning("llama.cpp update check failed (network unavailable or rate-limited); continuing")
        return

    current_str = f"b{info.current}" if info.current is not None else "unknown"

    if not info.available:
        log.info("llama.cpp is up to date (%s, latest b%s)", current_str, info.latest)
        return

    log.warning(
        "llama.cpp update available: %s → b%s  (%s)",
        current_str, info.latest, info.url,
    )

    if _already_updated:
        return

    if cfg.llama_update_auto:
        do_update = True
        log.info("LLAMA_UPDATE_AUTO=true — installing update automatically.")
    else:
        answer = await _ask_terminal(
            f"\nInstall llama.cpp b{info.latest}? [y/N] (auto-skip in "
            f"{cfg.llama_update_prompt_timeout_seconds}s): ",
            timeout=float(cfg.llama_update_prompt_timeout_seconds),
        )
        if answer is None:
            log.info("Update prompt timed out — continuing without updating.")
            return
        do_update = answer.strip().lower() in ("y", "yes")

    if not do_update:
        log.info("Update declined — continuing with current version.")
        return

    install_dir = Path(cfg.llama_server_executable).parent
    log.info("Installing llama.cpp b%s into %s ...", info.latest, install_dir)
    result = await download_and_install(
        install_dir,
        timeout_seconds=120.0,
    )

    if result.ok:
        log.info("llama.cpp updated successfully: %s", result.message)
        await _check_for_llama_update(_already_updated=True)
    else:
        log.warning("llama.cpp update failed: %s — continuing with current version.", result.message)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    init_db()
    bot = LocalBot()
    bot.run(cfg.discord_bot_token)
