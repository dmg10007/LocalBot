"""Per-user conversation history backed by SQLite.

All public functions are synchronous and protected by a threading.Lock.
The underlying connection is opened once (WAL mode) and reused for the
lifetime of the process.  async callers should use asyncio.to_thread().
"""
from __future__ import annotations

import sqlite3
import threading
from typing import TypedDict

from localbot.config import cfg


class Message(TypedDict):
    role: str
    content: str


_lock = threading.Lock()
_con: sqlite3.Connection | None = None


def _get_con() -> sqlite3.Connection:
    global _con
    if _con is None:
        _con = sqlite3.connect(cfg.database_path, check_same_thread=False)
        _con.execute("PRAGMA journal_mode=WAL")
        _con.execute("PRAGMA synchronous=NORMAL")
    return _con


def get_history(user_id: str) -> list[Message]:
    """Return the most-recent *max_history_messages* messages for *user_id*.

    Results are returned oldest-first so callers can append them directly
    to the LLM message list.
    """
    con = _get_con()
    with _lock:
        rows = con.execute(
            "SELECT role, content FROM history "
            "WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, cfg.max_history_messages),
        ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def append_message(user_id: str, role: str, content: str) -> None:
    """Insert *message* and atomically trim history to the configured cap.

    Uses a single DELETE ... NOT IN ... statement so no separate COUNT(*)
    round-trip is needed.
    """
    con = _get_con()
    with _lock:
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
    con = _get_con()
    with _lock:
        with con:
            con.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
