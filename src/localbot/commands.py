"""Discord slash-style text commands handled before agent routing.

Each command is a plain async function with a well-defined contract:

    async def handler(bot, message, user_id, text) -> bool

Returns True when the command matched and was handled; False otherwise.
Registering a new command is a single-line append to COMMAND_HANDLERS.
"""
from __future__ import annotations

import re
from typing import Callable, Awaitable, TYPE_CHECKING

import discord

from localbot.scheduler.store import get_user_timezone, set_user_timezone
from localbot.storage.history import clear_history
from localbot.tools.time_tools import get_current_time

if TYPE_CHECKING:
    from localbot.app import LocalBot

CommandHandler = Callable[["LocalBot", discord.Message, str, str], Awaitable[bool]]


async def _cmd_jobs_list(
    bot: "LocalBot", message: discord.Message, user_id: str, text: str
) -> bool:
    if text.lower() != "jobs list":
        return False
    jobs = bot.scheduler.list_user_jobs(user_id)
    if not jobs:
        await message.channel.send("You have no scheduled jobs.")
    else:
        lines = [f"`{j.job_id}` — `{j.cron_expr}` — {j.prompt}" for j in jobs]
        await message.channel.send("**Your scheduled jobs:**\n" + "\n".join(lines))
    return True


async def _cmd_jobs_cancel(
    bot: "LocalBot", message: discord.Message, user_id: str, text: str
) -> bool:
    m = re.match(r"^jobs cancel ([a-zA-Z0-9_-]+)$", text, re.IGNORECASE)
    if not m:
        return False
    cancelled = bot.scheduler.cancel_job(m.group(1))
    await message.channel.send(
        f"Job `{m.group(1)}` cancelled." if cancelled else "Job not found."
    )
    return True


async def _cmd_timezone_set(
    bot: "LocalBot", message: discord.Message, user_id: str, text: str
) -> bool:
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


async def _cmd_timezone_show(
    bot: "LocalBot", message: discord.Message, user_id: str, text: str
) -> bool:
    if text.lower() != "timezone show":
        return False
    tz = get_user_timezone(user_id)
    await message.channel.send(f"Your timezone is `{tz}`.")
    return True


async def _cmd_time_now(
    bot: "LocalBot", message: discord.Message, user_id: str, text: str
) -> bool:
    if text.lower() != "time now":
        return False
    tz = get_user_timezone(user_id)
    await message.channel.send(get_current_time(tz))
    return True


async def _cmd_clear(
    bot: "LocalBot", message: discord.Message, user_id: str, text: str
) -> bool:
    if text.lower() not in ("clear", "clear history", "/clear"):
        return False
    clear_history(user_id)
    await message.channel.send("Conversation history cleared.")
    return True


async def _cmd_model_status(
    bot: "LocalBot", message: discord.Message, user_id: str, text: str
) -> bool:
    if text.lower() not in ("model status", "model"):
        return False
    registry = bot.registry
    active = registry._active_slot or "none"  # noqa: SLF001
    lines = [
        f"  `{name}` — {'active' if name == active else ('available' if registry.is_slot_available(name) else 'not configured')}"  # type: ignore[arg-type]
        for name in ("general", "coding", "reasoning")
    ]
    await message.channel.send("**Model slots:**\n" + "\n".join(lines))
    return True


async def _cmd_help(
    bot: "LocalBot", message: discord.Message, user_id: str, text: str
) -> bool:
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


COMMAND_HANDLERS: list[CommandHandler] = [
    _cmd_jobs_list,
    _cmd_jobs_cancel,
    _cmd_timezone_set,
    _cmd_timezone_show,
    _cmd_time_now,
    _cmd_clear,
    _cmd_model_status,
    _cmd_help,
]
