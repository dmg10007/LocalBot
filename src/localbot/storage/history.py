"""Per-user conversation history backed by SQLite."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from typing import TypedDict

from localbot.config import cfg


class Message(TypedDict):
    role: str
    content: str


def _con() -> sqlite3.Connection:
    con = sqlite3.connect(cfg.database_path)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def get_history(user_id: str) -> list[Message]:
    # Fix #1: use closing() to guarantee the connection is released even on error.
    with closing(_con()) as con:
        rows = con.execute(
            "SELECT role, content FROM history WHERE user_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (user_id, cfg.max_history_messages),
        ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def append_message(user_id: str, role: str, content: str) -> None:
    # Fix #1 & #12: use closing() for safety; drop the COUNT(*) — the
    # DELETE … NOT IN … is a no-op when history is already within the cap,
    # so running it unconditionally is simpler and avoids an extra round-trip.
    with closing(_con()) as con:
        with con:
            con.execute(
                "INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)",
                (user_id, role, content),
            )
            con.execute(
                "DELETE FROM history WHERE user_id = ? AND id NOT IN "
                "(SELECT id FROM history WHERE user_id = ? ORDER BY id DESC LIMIT ?)",
                (user_id, user_id, cfg.max_history_messages),
            )


def clear_history(user_id: str) -> None:
    with closing(_con()) as con:
        with con:
            con.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
