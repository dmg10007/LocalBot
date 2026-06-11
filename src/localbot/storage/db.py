"""SQLite schema initialisation.

Call init_db() once at process startup before any storage reads/writes.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing

from localbot.config import cfg


def init_db() -> None:
    """Create all tables, indexes, and required directories.

    Idempotent: safe to call multiple times (all CREATE statements use
    IF NOT EXISTS).  The ALTER TABLE migration is caught and ignored when
    the column already exists.
    """
    os.makedirs(os.path.dirname(cfg.database_path) or "storage", exist_ok=True)
    os.makedirs(os.path.dirname(cfg.audit_log_path) or "logs", exist_ok=True)

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
                CREATE INDEX IF NOT EXISTS idx_history_user_id
                    ON history(user_id, id DESC);
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

        # Non-destructive migration for databases created before the
        # timezone column was added.
        try:
            con.execute(
                "ALTER TABLE scheduled_jobs "
                "ADD COLUMN timezone TEXT NOT NULL DEFAULT 'UTC'"
            )
            con.commit()
        except sqlite3.OperationalError:
            pass
