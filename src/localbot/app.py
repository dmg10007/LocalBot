"""Entry point, Discord client, and command handler."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import defaultdict

import discord

from localbot.adapters.llamacpp_client import LlamaCppClient
from localbot.adapters.llamacpp_server import LlamaCppServer
from localbot.agent import Agent
from localbot.config import cfg
from localbot.scheduler.service import SchedulerService
from localbot.scheduler.store import get_user_timezone, set_user_timezone
from localbot.storage.db import init_db
from localbot.storage.history import clear_history
from localbot.tools.time_tools import get_current_time
from localbot.messaging import split_message

log = logging.getLogger(__name__)


class LocalBot(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self._server = LlamaCppServer()
        self._client = LlamaCppClient()
        self._agent = Agent(self._server, self._client)
        self._scheduler = SchedulerService(self._send_scheduled)
        self._backend_ready = False
        # Per-user rate limiting: tracks last LLM request timestamp.
        self._last_request: dict[str, float] = defaultdict(float)

    # ------------------------------------------------------------------
    # Discord events
    # ------------------------------------------------------------------

    async def on_ready(self) -> None:
        log.info("Logged in as %s (id=%s)", self.user, self.user.id if self.user else "?")
        # Start backend in a background task so Discord events aren't blocked
        # during the ~30 s llama-server warmup.
        asyncio.create_task(self._start_backend())

    async def _start_backend(self) -> None:
        try:
            await self._server.start()
            await self._client.wait_until_ready()
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

        # Input length cap — reject oversized messages before they reach the LLM.
        if len(text) > cfg.max_input_length:
            await message.channel.send(
                f"Your message is too long (max {cfg.max_input_length} characters). Please shorten it."
            )
            return

        # Per-user rate limiting — prevent request flooding.
        now = time.monotonic()
        if now - self._last_request[user_id] < cfg.rate_limit_seconds:
            remaining = cfg.rate_limit_seconds - (now - self._last_request[user_id])
            await message.channel.send(
                f"Please wait {remaining:.1f}s before sending another message."
            )
            return
        self._last_request[user_id] = now

        async with message.channel.typing():
            reply = await self._agent.handle(user_id, text)

        for chunk in split_message(reply):
            await message.channel.send(chunk)

    async def close(self) -> None:
        self._scheduler.stop()
        await self._client.close()
        await self._server.stop()
        await super().close()

    # ------------------------------------------------------------------
    # Built-in command handler
    # ------------------------------------------------------------------

    async def _handle_command(self, message: discord.Message, user_id: str, text: str) -> bool:
        """Return True if the message was a built-in command."""
        lower = text.lower()

        if lower == "jobs list":
            jobs = self._scheduler.list_user_jobs(user_id)
            if not jobs:
                await message.channel.send("You have no scheduled jobs.")
            else:
                lines = [f"`{j.job_id}` — `{j.cron_expr}` — {j.prompt}" for j in jobs]
                await message.channel.send("**Your scheduled jobs:**\n" + "\n".join(lines))
            return True

        # jobs cancel <id> — match on original text, case-insensitive, flexible ID format
        m = re.match(r"^jobs cancel ([a-zA-Z0-9_-]+)$", text, re.IGNORECASE)
        if m:
            cancelled = self._scheduler.cancel_job(m.group(1))
            await message.channel.send(
                f"Job `{m.group(1)}` cancelled." if cancelled else "Job not found."
            )
            return True

        m = re.match(r"^timezone set (.+)$", text, re.IGNORECASE)
        if m:
            tz = m.group(1).strip()
            set_user_timezone(user_id, tz)
            await message.channel.send(f"Timezone set to `{tz}`.")
            return True

        if lower == "timezone show":
            tz = get_user_timezone(user_id)
            await message.channel.send(f"Your timezone is `{tz}`.")
            return True

        if lower == "time now":
            tz = get_user_timezone(user_id)
            await message.channel.send(get_current_time(tz))
            return True

        if lower in ("clear", "clear history", "/clear"):
            clear_history(user_id)
            await message.channel.send("Conversation history cleared.")
            return True

        if lower in ("help", "/help"):
            await message.channel.send(
                "**LocalBot commands**\n"
                "`jobs list` — List your scheduled jobs\n"
                "`jobs cancel <id>` — Cancel a scheduled job\n"
                "`timezone set <IANA>` — Set your timezone (e.g. `America/New_York`)\n"
                "`timezone show` — Show your current timezone\n"
                "`time now` — Show current time in your timezone\n"
                "`clear` — Clear your conversation history\n"
                "\nFor scheduled reminders, just ask naturally:\n"
                "> *Remind me every morning at 8am to review my task list*"
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Scheduler callback
    # ------------------------------------------------------------------

    async def _send_scheduled(self, user_id: str, prompt: str) -> None:
        """Called by the scheduler to deliver a prompt to a user via DM."""
        try:
            discord_user = await self.fetch_user(int(user_id))
            dm = await discord_user.create_dm()
            async with dm.typing():
                reply = await self._agent.handle(user_id, prompt)
            for chunk in split_message(reply):
                await dm.send(chunk)
        except Exception:
            log.exception("Failed to deliver scheduled message to user %s", user_id)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    init_db()
    bot = LocalBot()
    bot.run(cfg.discord_bot_token)
