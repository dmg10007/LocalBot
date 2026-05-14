"""SQLite schema initialisation."""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing

from localbot.config import cfg


def init_db() -> None:
    """Create tables, indexes, and required directories if they don't exist."""
    os.makedirs(os.path.dirname(cfg.database_path) or "storage", exist_ok=True)
    os.makedirs(os.path.dirname(cfg.audit_log_path) or "logs", exist_ok=True)

    # Fix #3: use contextlib.closing so con.close() is always called, even
    # when executescript or the ALTER TABLE migration raises an exception.
    with closing(sqlite3.connect(cfg.database_path)) as con:
        with con:
            con.executescript("""
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS history (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id   TEXT    NOT NULL,
                    role      TEXT    NOT NULL,
                    content   TEXT    NOT NULL,
                    ts        REAL    NOT NULL DEFAULT (unixepoch('now'))
                );

                -- Primary index for per-user history lookups ordered by insertion.
                CREATE INDEX IF NOT EXISTS idx_history_user_id
                    ON history(user_id, id DESC);

                -- Keep the ts-based index for any range queries on timestamps.
                CREATE INDEX IF NOT EXISTS idx_history_user_ts
                    ON history(user_id, ts DESC);

                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id   TEXT PRIMARY KEY,
                    timezone  TEXT NOT NULL DEFAULT 'UTC'
                );

                CREATE TABLE IF NOT EXISTS scheduled_jobs (
                    job_id      TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    prompt      TEXT NOT NULL,
                    cron_expr   TEXT NOT NULL,
                    timezone    TEXT NOT NULL DEFAULT 'UTC',
                    created_at  REAL NOT NULL DEFAULT (unixepoch('now'))
                );
            """)

        # Non-destructive migration: add timezone column to existing databases
        # that were created before this column existed.  ALTER TABLE ADD COLUMN
        # is a no-op if the column is already present (caught below).
        # Note: this runs outside the `with con:` transaction block because
        # DDL inside a transaction can behave unexpectedly in some SQLite versions.
        try:
            con.execute(
                "ALTER TABLE scheduled_jobs ADD COLUMN timezone TEXT NOT NULL DEFAULT 'UTC'"
            )
            con.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
