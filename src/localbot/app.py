"""Entry point, Discord client, and command handler."""
from __future__ import annotations

import asyncio
import logging
import os
import re

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

    # ------------------------------------------------------------------
    # Discord events
    # ------------------------------------------------------------------

    async def on_ready(self) -> None:
        log.info("Logged in as %s (id=%s)", self.user, self.user.id if self.user else "?")
        await self._server.start()
        await self._client.wait_until_ready()
        self._scheduler.start()

    async def on_message(self, message: discord.Message) -> None:
        # Ignore bots and non-DM messages
        if message.author.bot:
            return
        if not isinstance(message.channel, discord.DMChannel):
            return

        user_id = str(message.author.id)
        text = message.content.strip()

        # Handle built-in commands first
        if await self._handle_command(message, user_id, text):
            return

        # Otherwise run through the agent
        async with message.channel.typing():
            reply = await self._agent.handle(user_id, text)

        for chunk in split_message(reply):
            await message.channel.send(chunk)

    async def close(self) -> None:
        self._scheduler.stop()
        await self._server.stop()
        await super().close()

    # ------------------------------------------------------------------
    # Built-in command handler
    # ------------------------------------------------------------------

    async def _handle_command(self, message: discord.Message, user_id: str, text: str) -> bool:
        """Return True if the message was a built-in command."""
        lower = text.lower()

        # jobs list
        if lower == "jobs list":
            jobs = self._scheduler.list_user_jobs(user_id)
            if not jobs:
                await message.channel.send("You have no scheduled jobs.")
            else:
                lines = [f"`{j.job_id}` — `{j.cron_expr}` — {j.prompt}" for j in jobs]
                await message.channel.send("**Your scheduled jobs:**\n" + "\n".join(lines))
            return True

        # jobs cancel <id>
        m = re.match(r"^jobs cancel ([a-z0-9]+)$", lower)
        if m:
            cancelled = self._scheduler.cancel_job(m.group(1))
            await message.channel.send(
                f"Job `{m.group(1)}` cancelled." if cancelled else "Job not found."
            )
            return True

        # timezone set <tz>
        m = re.match(r"^timezone set (.+)$", text, re.IGNORECASE)
        if m:
            tz = m.group(1).strip()
            set_user_timezone(user_id, tz)
            await message.channel.send(f"Timezone set to `{tz}`.")
            return True

        # timezone show
        if lower == "timezone show":
            tz = get_user_timezone(user_id)
            await message.channel.send(f"Your timezone is `{tz}`.")
            return True

        # time now
        if lower == "time now":
            tz = get_user_timezone(user_id)
            await message.channel.send(get_current_time(tz))
            return True

        # clear history
        if lower in ("clear", "clear history", "/clear"):
            clear_history(user_id)
            await message.channel.send("Conversation history cleared.")
            return True

        # help
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
        """Called by the scheduler to deliver a scheduled prompt to a user."""
        try:
            user = await self.fetch_user(int(user_id))
            dm = await user.create_dm()
            reply = await self._agent.handle(user_id, prompt)
            for chunk in split_message(reply):
                await dm.send(chunk)
        except Exception as exc:
            log.error("Failed to send scheduled message to %s: %s", user_id, exc)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not cfg.discord_bot_token:
        raise SystemExit("DISCORD_BOT_TOKEN is not set in .env")
    if not cfg.llama_server_model_path:
        raise SystemExit("LLAMA_SERVER_MODEL_PATH is not set in .env")

    init_db()
    bot = LocalBot()
    bot.run(cfg.discord_bot_token)
