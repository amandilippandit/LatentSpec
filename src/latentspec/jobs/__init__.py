"""Background job runner — async mining without blocking the API.

A job runner that:
  - persists job state to the `mining_jobs` table
  - executes registered handlers in a background asyncio task pool
  - reports progress via `progress_percent` + `progress_message` updates
  - survives partial failures (one job's exception doesn't kill the runner)

Production deploys can swap this for Celery by replacing
`InProcessJobRunner` with a Celery-backed implementation that satisfies
the same `JobRunner` protocol — every API caller goes through the
runner, never imports a Celery primitive directly.
"""

from latentspec.jobs.runner import (
    InProcessJobRunner,
    JobHandler,
    JobRunner,
    enqueue_job,
    get_runner,
    register_handler,
)

__all__ = [
    "InProcessJobRunner",
    "JobHandler",
    "JobRunner",
    "enqueue_job",
    "get_runner",
    "register_handler",
]
