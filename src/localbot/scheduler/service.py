"""APScheduler wrapper for scheduled prompts."""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Callable, Awaitable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from localbot.config import cfg
from localbot.scheduler.store import Job, save_job, delete_job, list_jobs, all_jobs

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

SendCallback = Callable[[str, str], Awaitable[None]]


class SchedulerService:
    def __init__(self, send_cb: SendCallback) -> None:
        self._send = send_cb
        self._scheduler = AsyncIOScheduler()

    def start(self) -> None:
        self._scheduler.start()
        persisted = all_jobs()  # single DB call
        for job in persisted:
            self._register(job)
        log.info("Scheduler started with %d persisted jobs", len(persisted))

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        log.info("Scheduler stopped")

    def _register(self, job: Job) -> None:
        parts = job.cron_expr.split()
        if len(parts) != 5:
            log.warning("Invalid cron expression for job %s: %r", job.job_id, job.cron_expr)
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
        total = len(all_jobs())
        user_total = len(list_jobs(user_id))
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
