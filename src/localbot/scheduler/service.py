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

# Callback type: receives (user_id, prompt) and sends the reply
SendCallback = Callable[[str, str], Awaitable[None]]


class SchedulerService:
    def __init__(self, send_cb: SendCallback) -> None:
        self._send = send_cb
        self._scheduler = AsyncIOScheduler()

    def start(self) -> None:
        self._scheduler.start()
        # Re-register persisted jobs on startup
        for job in all_jobs():
            self._register(job)
        log.info("Scheduler started with %d persisted jobs", len(all_jobs()))

    def stop(self) -> None:
        self._scheduler.shutdown(wait=False)

    def _register(self, job: Job) -> None:
        async def _run() -> None:
            await self._send(job.user_id, job.prompt)

        try:
            trigger = CronTrigger.from_crontab(job.cron_expr)
            self._scheduler.add_job(
                _run,
                trigger=trigger,
                id=job.job_id,
                replace_existing=True,
                max_instances=1,
            )
        except Exception as exc:
            log.error("Failed to register job %s: %s", job.job_id, exc)

    def add_job(self, user_id: str, prompt: str, cron_expr: str) -> Job:
        job = Job(
            job_id=str(uuid.uuid4())[:8],
            user_id=user_id,
            prompt=prompt,
            cron_expr=cron_expr,
        )
        save_job(job)
        self._register(job)
        log.info("Scheduled job %s for user %s: %s", job.job_id, user_id, cron_expr)
        return job

    def cancel_job(self, job_id: str) -> bool:
        try:
            self._scheduler.remove_job(job_id)
        except Exception:
            pass
        delete_job(job_id)
        return True

    def list_user_jobs(self, user_id: str) -> list[Job]:
        return list_jobs(user_id)

    def job_count_for_user(self, user_id: str) -> int:
        return len(list_jobs(user_id))
