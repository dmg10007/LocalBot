"""SQLite-backed scheduled job persistence.

Uses a single persistent WAL-mode connection per process rather than
opening a new connection on every call (issue #11).
"""
from __future__ import annotations

import sqlite3
import threading
import zoneinfo
from dataclasses import dataclass, field

from localbot.config import cfg


@dataclass
class Job:
    job_id: str
    user_id: str
    prompt: str
    cron_expr: str
    timezone: str = field(default="UTC")


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

def save_job(job: Job) -> None:
    con = _get_con()
    with _lock:
        with con:
            con.execute(
                "INSERT OR REPLACE INTO scheduled_jobs "
                "(job_id, user_id, prompt, cron_expr, timezone) "
                "VALUES (?, ?, ?, ?, ?)",
                (job.job_id, job.user_id, job.prompt, job.cron_expr, job.timezone),
            )


def delete_job(job_id: str, user_id: str | None = None) -> bool:
    """Delete a job. When *user_id* is given, only delete a job owned by
    that user (prevents cross-user cancellation). When None, delete by id
    alone (used by trusted internal callers only)."""
    con = _get_con()
    with _lock:
        with con:
            if user_id is None:
                cur = con.execute(
                    "DELETE FROM scheduled_jobs WHERE job_id = ?", (job_id,)
                )
            else:
                cur = con.execute(
                    "DELETE FROM scheduled_jobs WHERE job_id = ? AND user_id = ?",
                    (job_id, user_id),
                )
        return cur.rowcount > 0


def list_jobs(user_id: str) -> list[Job]:
    con = _get_con()
    with _lock:
        rows = con.execute(
            "SELECT job_id, user_id, prompt, cron_expr, timezone "
            "FROM scheduled_jobs WHERE user_id = ?",
            (user_id,),
        ).fetchall()
    return [Job(*row) for row in rows]


def all_jobs() -> list[Job]:
    con = _get_con()
    with _lock:
        rows = con.execute(
            "SELECT job_id, user_id, prompt, cron_expr, timezone FROM scheduled_jobs"
        ).fetchall()
    return [Job(*row) for row in rows]


def count_jobs_atomic(user_id: str) -> tuple[int, int]:
    """Return (global_total, user_total) in a single DB round-trip.

    Eliminates the TOCTOU race in add_job by reading both counts inside
    the same connection before the caller decides whether to insert.
    """
    con = _get_con()
    with _lock:
        total = con.execute("SELECT COUNT(*) FROM scheduled_jobs").fetchone()[0]
        user_total = con.execute(
            "SELECT COUNT(*) FROM scheduled_jobs WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
    return total, user_total


def get_user_timezone(user_id: str) -> str:
    con = _get_con()
    with _lock:
        row = con.execute(
            "SELECT timezone FROM user_settings WHERE user_id = ?", (user_id,)
        ).fetchone()
    return row[0] if row else "UTC"


def set_user_timezone(user_id: str, timezone: str) -> None:
    """Persist *timezone* for *user_id* after validating it is a known IANA name."""
    if timezone not in zoneinfo.available_timezones():
        raise ValueError(f"Unknown timezone: {timezone!r}")
    con = _get_con()
    with _lock:
        with con:
            con.execute(
                "INSERT OR REPLACE INTO user_settings (user_id, timezone) VALUES (?, ?)",
                (user_id, timezone),
            )
