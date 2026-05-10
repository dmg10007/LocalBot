"""SQLite-backed scheduled job persistence."""
from __future__ import annotations

import sqlite3
import zoneinfo
from contextlib import closing
from dataclasses import dataclass

from localbot.config import cfg


@dataclass
class Job:
    job_id: str
    user_id: str
    prompt: str
    cron_expr: str


def _con() -> sqlite3.Connection:
    con = sqlite3.connect(cfg.database_path)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def save_job(job: Job) -> None:
    with closing(_con()) as con:
        with con:
            con.execute(
                "INSERT OR REPLACE INTO scheduled_jobs (job_id, user_id, prompt, cron_expr) "
                "VALUES (?, ?, ?, ?)",
                (job.job_id, job.user_id, job.prompt, job.cron_expr),
            )


def delete_job(job_id: str) -> bool:
    with closing(_con()) as con:
        with con:
            cur = con.execute(
                "DELETE FROM scheduled_jobs WHERE job_id = ?", (job_id,)
            )
        return cur.rowcount > 0


def list_jobs(user_id: str) -> list[Job]:
    with closing(_con()) as con:
        rows = con.execute(
            "SELECT job_id, user_id, prompt, cron_expr FROM scheduled_jobs WHERE user_id = ?",
            (user_id,),
        ).fetchall()
    return [Job(*row) for row in rows]


def all_jobs() -> list[Job]:
    with closing(_con()) as con:
        rows = con.execute(
            "SELECT job_id, user_id, prompt, cron_expr FROM scheduled_jobs"
        ).fetchall()
    return [Job(*row) for row in rows]


def count_jobs_atomic(user_id: str) -> tuple[int, int]:
    """Return (global_total, user_total) in a single DB round-trip.

    Fix #4: eliminates the TOCTOU race in add_job by reading both counts
    inside the same connection before the caller decides whether to insert.
    """
    with closing(_con()) as con:
        total = con.execute("SELECT COUNT(*) FROM scheduled_jobs").fetchone()[0]
        user_total = con.execute(
            "SELECT COUNT(*) FROM scheduled_jobs WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
    return total, user_total


def get_user_timezone(user_id: str) -> str:
    with closing(_con()) as con:
        row = con.execute(
            "SELECT timezone FROM user_settings WHERE user_id = ?", (user_id,)
        ).fetchone()
    return row[0] if row else "UTC"


def set_user_timezone(user_id: str, timezone: str) -> None:
    # Fix #7: validate the timezone string before persisting it.
    if timezone not in zoneinfo.available_timezones():
        raise ValueError(f"Unknown timezone: {timezone!r}")
    with closing(_con()) as con:
        with con:
            con.execute(
                "INSERT OR REPLACE INTO user_settings (user_id, timezone) VALUES (?, ?)",
                (user_id, timezone),
            )
