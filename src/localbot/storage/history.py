"""Per-user conversation history backed by SQLite."""
from __future__ import annotations

import sqlite3
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
    con = _con()
    rows = con.execute(
        "SELECT role, content FROM history WHERE user_id = ? "
        "ORDER BY ts DESC LIMIT ?",
        (user_id, cfg.max_history_messages),
    ).fetchall()
    con.close()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def append_message(user_id: str, role: str, content: str) -> None:
    con = _con()
    with con:
        con.execute(
            "INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content),
        )
        # Trim oldest rows beyond the cap using the index on (user_id, ts DESC)
        con.execute(
            "DELETE FROM history WHERE user_id = ? AND id NOT IN "
            "(SELECT id FROM history WHERE user_id = ? ORDER BY ts DESC LIMIT ?)",
            (user_id, user_id, cfg.max_history_messages),
        )
    con.close()


def clear_history(user_id: str) -> None:
    con = _con()
    with con:
        con.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
    con.close()
