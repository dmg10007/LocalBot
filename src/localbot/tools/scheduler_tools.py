"""LLM-callable wrappers around SchedulerService.

This module exposes three thin async functions — schedule_job, cancel_job,
and list_jobs — that the agent's tool loop can dispatch to.  They are
not registered globally; instead a SchedulerTools instance is constructed
in app.py and injected into Agent so the live SchedulerService reference
is available without a global singleton.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from localbot.scheduler.service import SchedulerService

log = logging.getLogger(__name__)

# OpenAI-style schemas for the three scheduler tools.
SCHEDULER_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "schedule_job",
            "description": (
                "Create a recurring scheduled message for the user. "
                "Call this when the user asks to be reminded, notified, or sent "
                "a message on a recurring schedule. "
                "cron_expr must be a standard 5-field cron string "
                "(minute hour day month day_of_week). "
                "Examples: every day at 8 AM = '0 8 * * *', "
                "every Monday at 9 AM = '0 9 * * 1', "
                "every hour = '0 * * * *'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The message or prompt to deliver to the user when the job fires.",
                    },
                    "cron_expr": {
                        "type": "string",
                        "description": "5-field cron expression, e.g. '0 8 * * *' for 8 AM every day.",
                    },
                },
                "required": ["prompt", "cron_expr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_job",
            "description": (
                "Cancel an existing scheduled job by its ID. "
                "Use list_jobs first if you need to find the job ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID to cancel (8-character hex string).",
                    }
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_jobs",
            "description": (
                "List all active scheduled jobs for the current user. "
                "Use this when the user asks what jobs are scheduled or "
                "needs a job ID to cancel one."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


class SchedulerTools:
    """Holds a reference to the live SchedulerService and a user_id context.

    A new instance is created per-request inside Agent.handle() so that
    each tool call automatically targets the correct user.
    """

    def __init__(self, service: "SchedulerService", user_id: str) -> None:
        self._service = service
        self._user_id = user_id

    async def schedule_job(self, prompt: str, cron_expr: str) -> str:
        try:
            job = self._service.add_job(
                user_id=self._user_id,
                prompt=prompt,
                cron_expr=cron_expr,
            )
            log.info(
                "Scheduled job %s for user %s: cron=%r prompt=%r",
                job.job_id, self._user_id, cron_expr, prompt,
            )
            return (
                f"Job scheduled successfully. "
                f"ID: `{job.job_id}`, schedule: `{cron_expr}`. "
                f"The user can cancel it with `jobs cancel {job.job_id}`."
            )
        except ValueError as exc:
            return f"Could not schedule job: {exc}"
        except Exception as exc:
            log.exception("Unexpected error scheduling job for user %s", self._user_id)
            return f"Unexpected error scheduling job: {exc}"

    async def cancel_job(self, job_id: str) -> str:
        cancelled = self._service.cancel_job(job_id)
        if cancelled:
            log.info("Cancelled job %s for user %s", job_id, self._user_id)
            return f"Job `{job_id}` has been cancelled."
        return f"No job found with ID `{job_id}`."

    async def list_jobs(self) -> str:
        jobs = self._service.list_user_jobs(self._user_id)
        if not jobs:
            return "No scheduled jobs found for this user."
        lines = [
            f"- `{j.job_id}` | `{j.cron_expr}` | {j.prompt}"
            for j in jobs
        ]
        return "Active scheduled jobs:\n" + "\n".join(lines)

    async def dispatch(self, tool_name: str, args: dict[str, Any]) -> str | None:
        """Dispatch a scheduler tool call. Returns None if tool_name is not ours."""
        if tool_name == "schedule_job":
            return await self.schedule_job(
                prompt=args["prompt"],
                cron_expr=args["cron_expr"],
            )
        if tool_name == "cancel_job":
            return await self.cancel_job(job_id=args["job_id"])
        if tool_name == "list_jobs":
            return await self.list_jobs()
        return None
