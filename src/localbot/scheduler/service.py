"""APScheduler wrapper for scheduled prompts."""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
from typing import Callable, Awaitable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from localbot.config import cfg
from localbot.scheduler.store import Job, save_job, delete_job, list_jobs, all_jobs, count_jobs_atomic

log = logging.getLogger(__name__)

SendCallback = Callable[[str, str], Awaitable[None]]

# Fix #6: validate each cron field is within its legal range before
# passing it to APScheduler. Supports plain values and */step syntax.
_CRON_FIELD_PATTERNS = [
    ("minute",      re.compile(r"^(\*|([0-9]|[1-5][0-9]))(/(\d+))?$")),
    ("hour",        re.compile(r"^(\*|([0-9]|1[0-9]|2[0-3]))(/(\d+))?$")),
    ("day",         re.compile(r"^(\*|([1-9]|[12][0-9]|3[01]))(/(\d+))?$")),
    ("month",       re.compile(r"^(\*|([1-9]|1[0-2]))(/(\d+))?$")),
    ("day_of_week", re.compile(r"^(\*|[0-6])(/(\d+))?$")),
]


def _validate_cron(expr: str) -> str | None:
    """Return an error description if *expr* is invalid, else None."""
    parts = expr.split()
    if len(parts) != 5:
        return f"Expected 5 fields, got {len(parts)}"
    for (name, pattern), value in zip(_CRON_FIELD_PATTERNS, parts):
        if not pattern.fullmatch(value):
            return f"Invalid {name} field: {value!r}"
    return None


class SchedulerService:
    def __init__(self, send_cb: SendCallback) -> None:
        self._send = send_cb
        self._scheduler = AsyncIOScheduler()

    def start(self) -> None:
        # Fix #19: attach an error listener so job failures surface in our logs.
        self._scheduler.add_listener(self._on_job_error, mask=0x8000)  # EVENT_JOB_ERROR
        self._scheduler.start()
        persisted = all_jobs()
        for job in persisted:
            self._register(job)
        log.info("Scheduler started with %d persisted jobs", len(persisted))

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        log.info("Scheduler stopped")

    def _on_job_error(self, event) -> None:  # type: ignore[no-untyped-def]
        log.exception(
            "Scheduled job %s raised an exception: %s",
            event.job_id,
            event.exception,
            exc_info=event.traceback,
        )

    def _register(self, job: Job) -> None:
        parts = job.cron_expr.split()
        if len(parts) != 5:
            log.warning("Invalid cron expression for job %s: %r — skipping", job.job_id, job.cron_expr)
            return
        minute, hour, day, month, day_of_week = parts
        self._scheduler.add_job(
            self._fire,
            CronTrigger(
                minute=minute, hour=hour, day=day,
                month=month, day_of_week=day_of_week,
            ),
            args=[job.user_id, job.prompt],
            id=job.job_id,
            replace_existing=True,
        )

    async def _fire(self, user_id: str, prompt: str) -> None:
        await self._send(user_id, prompt)

    def add_job(self, user_id: str, prompt: str, cron_expr: str) -> Job:
        # Fix #6: reject invalid cron expressions before storing.
        err = _validate_cron(cron_expr)
        if err:
            raise ValueError(f"Invalid cron expression {cron_expr!r}: {err}")

        # Fix #4: single atomic DB read to eliminate the TOCTOU race.
        total, user_total = count_jobs_atomic(user_id)
        if total >= cfg.scheduler_max_jobs:
            raise ValueError("Global job limit reached.")
        if user_total >= cfg.scheduler_max_jobs_per_user:
            raise ValueError("Per-user job limit reached.")

        job = Job(
            job_id=uuid.uuid4().hex[:8],
            user_id=user_id,
            prompt=prompt,
            cron_expr=cron_expr,
        )
        save_job(job)
        self._register(job)
        return job

    def cancel_job(self, job_id: str) -> bool:
        try:
            self._scheduler.remove_job(job_id)
        except Exception:
            pass
        return delete_job(job_id)

    def list_user_jobs(self, user_id: str) -> list[Job]:
        return list_jobs(user_id)
