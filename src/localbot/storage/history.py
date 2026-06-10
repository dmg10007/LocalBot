"""Per-user conversation history backed by SQLite.

Uses a single persistent WAL-mode connection per process rather than
opening a new connection on every call (issue #11).
"""
from __future__ import annotations

import sqlite3
import threading
from typing import TypedDict

from localbot.config import cfg


class Message(TypedDict):
    role: str
    content: str


# ---------------------------------------------------------------------------
# Shared connection
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_con: sqlite3.Connection | None = None


def _get_con() -> sqlite3.Connection:
    """Return the module-level connection, creating it on first call."""
    global _con
    if _con is None:
        _con = sqlite3.connect(
            cfg.database_path,
            check_same_thread=False,  # guarded by _lock for writes
        )
        _con.execute("PRAGMA journal_mode=WAL")
        _con.execute("PRAGMA synchronous=NORMAL")
    return _con


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_history(user_id: str) -> list[Message]:
    con = _get_con()
    with _lock:
        rows = con.execute(
            "SELECT role, content FROM history WHERE user_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (user_id, cfg.max_history_messages),
        ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def append_message(user_id: str, role: str, content: str) -> None:
    """Insert a message and trim history to the configured cap.

    The DELETE ... NOT IN ... is a no-op when the row count is already
    within the cap, so no separate COUNT(*) round-trip is needed.
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
