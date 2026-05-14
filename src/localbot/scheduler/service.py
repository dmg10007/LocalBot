"""APScheduler wrapper for scheduled prompts."""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
import zoneinfo
from typing import Callable, Awaitable

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from localbot.config import cfg
from localbot.scheduler.store import (
    Job, save_job, delete_job, list_jobs, all_jobs, count_jobs_atomic,
    get_user_timezone,
)

log = logging.getLogger(__name__)

SendCallback = Callable[[str, str], Awaitable[None]]

# Validate each cron field is within its legal range before passing it
# to APScheduler.  Supports plain values, ranges (1-5), and */step syntax.
_CRON_FIELD_PATTERNS = [
    ("minute",      re.compile(r"^(\*|(([0-9]|[1-5][0-9])(-([0-9]|[1-5][0-9]))?)(/(\d+))?|\*/(\d+))$")),
    ("hour",        re.compile(r"^(\*|(([0-9]|1[0-9]|2[0-3])(-([0-9]|1[0-9]|2[0-3]))?)(/(\d+))?|\*/(\d+))$")),
    ("day",         re.compile(r"^(\*|(([1-9]|[12][0-9]|3[01])(-([1-9]|[12][0-9]|3[01]))?)(/(\d+))?|\*/(\d+))$")),
    ("month",       re.compile(r"^(\*|(([1-9]|1[0-2])(-([1-9]|1[0-2]))?)(/(\d+))?|\*/(\d+))$")),
    ("day_of_week", re.compile(r"^(\*|([0-6](-[0-6])?)(/(\d+))?|\*/(\d+))$")),
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
        # Use imported constants instead of magic numbers.
        # Also listen for EVENT_JOB_MISSED so silent misfires are logged.
        self._scheduler.add_listener(
            self._on_job_event,
            EVENT_JOB_ERROR | EVENT_JOB_MISSED,
        )
        self._scheduler.start()
        persisted = all_jobs()
        for job in persisted:
            self._register(job)
        log.info("Scheduler started with %d persisted jobs", len(persisted))

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        log.info("Scheduler stopped")

    def _on_job_event(self, event) -> None:  # type: ignore[no-untyped-def]
        if hasattr(event, "exception") and event.exception:
            log.error(
                "Scheduled job %s raised an exception: %s",
                event.job_id,
                event.exception,
                exc_info=event.traceback,
            )
        else:
            log.warning(
                "Scheduled job %s was missed (scheduler overloaded or bot was down)",
                event.job_id,
            )

    def _register(self, job: Job) -> None:
        parts = job.cron_expr.split()
        if len(parts) != 5:
            log.warning(
                "Skipping job %s — invalid cron expression: %r",
                job.job_id, job.cron_expr,
            )
            return
        minute, hour, day, month, day_of_week = parts

        # Resolve the timezone stored on the job (defaulting to UTC).
        try:
            tz = zoneinfo.ZoneInfo(job.timezone)
        except (zoneinfo.ZoneInfoNotFoundError, KeyError):
            log.warning(
                "Job %s has unknown timezone %r — falling back to UTC",
                job.job_id, job.timezone,
            )
            tz = zoneinfo.ZoneInfo("UTC")

        # Bug fix #1: APScheduler 3.x does not reliably await async bound
        # methods — asyncio.iscoroutinefunction() returns False for them in
        # several Python/APScheduler version combos, so the coroutine was
        # never awaited and the job silently did nothing.
        #
        # Fix: register a plain synchronous callback (_fire_sync) that
        # schedules the coroutine on the running event loop via create_task.
        self._scheduler.add_job(
            self._fire_sync,
            CronTrigger(
                minute=minute,
                hour=hour,
                day=day,
                month=month,
                day_of_week=day_of_week,
                timezone=tz,
            ),
            args=[job.user_id, job.prompt],
            id=job.job_id,
            replace_existing=True,
        )
        log.debug(
            "Registered job %s: cron=%r tz=%s",
            job.job_id, job.cron_expr, job.timezone,
        )

    def _fire_sync(self, user_id: str, prompt: str) -> None:
        """Synchronous APScheduler callback.

        Schedules _fire on the running event loop via create_task so the
        coroutine is properly awaited without relying on APScheduler's
        async detection of bound methods.
        """
        loop = asyncio.get_event_loop()
        loop.create_task(
            self._fire(user_id, prompt),
            name=f"scheduler-fire-{user_id}",
        )

    async def _fire(self, user_id: str, prompt: str) -> None:
        try:
            await self._send(user_id, prompt)
        except Exception:
            log.exception(
                "Unhandled exception delivering scheduled message to user %s",
                user_id,
            )

    def add_job(self, user_id: str, prompt: str, cron_expr: str) -> Job:
        err = _validate_cron(cron_expr)
        if err:
            raise ValueError(f"Invalid cron expression {cron_expr!r}: {err}")

        # Single atomic DB read to eliminate the TOCTOU race.
        total, user_total = count_jobs_atomic(user_id)
        if total >= cfg.scheduler_max_jobs:
            raise ValueError("Global job limit reached.")
        if user_total >= cfg.scheduler_max_jobs_per_user:
            raise ValueError("Per-user job limit reached.")

        # Bug fix #4: snapshot the user's timezone at creation time so that
        # later changes to user_settings do not silently shift existing jobs,
        # and so restarts re-register with the originally-intended timezone.
        timezone = get_user_timezone(user_id)

        job = Job(
            job_id=uuid.uuid4().hex[:8],
            user_id=user_id,
            prompt=prompt,
            cron_expr=cron_expr,
            timezone=timezone,
        )
        save_job(job)
        self._register(job)
        log.info(
            "Added job %s for user %s: cron=%r tz=%s prompt=%r",
            job.job_id, user_id, cron_expr, timezone, prompt,
        )
        return job

    def cancel_job(self, job_id: str) -> bool:
        try:
            self._scheduler.remove_job(job_id)
        except Exception:
            pass
        return delete_job(job_id)

    def list_user_jobs(self, user_id: str) -> list[Job]:
        return list_jobs(user_id)
