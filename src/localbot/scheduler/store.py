"""SQLite-backed scheduled job persistence."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from localbot.config import cfg


@dataclass
class Job:
    job_id: str
    user_id: str
    prompt: str
    cron_expr: str


def _con() -> sqlite3.Connection:
    return sqlite3.connect(cfg.database_path)


def save_job(job: Job) -> None:
    con = _con()
    with con:
        con.execute(
            "INSERT OR REPLACE INTO scheduled_jobs (job_id, user_id, prompt, cron_expr) "
            "VALUES (?, ?, ?, ?)",
            (job.job_id, job.user_id, job.prompt, job.cron_expr),
        )
    con.close()


def delete_job(job_id: str) -> None:
    con = _con()
    with con:
        con.execute("DELETE FROM scheduled_jobs WHERE job_id = ?", (job_id,))
    con.close()


def list_jobs(user_id: str) -> list[Job]:
    con = _con()
    rows = con.execute(
        "SELECT job_id, user_id, prompt, cron_expr FROM scheduled_jobs WHERE user_id = ?",
        (user_id,),
    ).fetchall()
    con.close()
    return [Job(*row) for row in rows]


def all_jobs() -> list[Job]:
    con = _con()
    rows = con.execute(
        "SELECT job_id, user_id, prompt, cron_expr FROM scheduled_jobs"
    ).fetchall()
    con.close()
    return [Job(*row) for row in rows]


def get_user_timezone(user_id: str) -> str:
    con = _con()
    row = con.execute(
        "SELECT timezone FROM user_settings WHERE user_id = ?", (user_id,)
    ).fetchone()
    con.close()
    return row[0] if row else "UTC"


def set_user_timezone(user_id: str, timezone: str) -> None:
    con = _con()
    with con:
        con.execute(
            "INSERT OR REPLACE INTO user_settings (user_id, timezone) VALUES (?, ?)",
            (user_id, timezone),
        )
    con.close()
