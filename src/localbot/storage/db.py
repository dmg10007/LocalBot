"""SQLite schema initialisation."""
from __future__ import annotations

import os
import sqlite3

from localbot.config import cfg


def init_db() -> None:
    """Create tables, indexes, and required directories if they don't exist."""
    os.makedirs(os.path.dirname(cfg.database_path) or "storage", exist_ok=True)
    os.makedirs(os.path.dirname(cfg.audit_log_path) or "logs", exist_ok=True)

    con = sqlite3.connect(cfg.database_path)
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

            -- Index for fast per-user history lookups and trim queries
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
                created_at  REAL NOT NULL DEFAULT (unixepoch('now'))
            );
        """)
    con.close()
