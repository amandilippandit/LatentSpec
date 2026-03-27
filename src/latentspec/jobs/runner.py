"""In-process async job runner with persistent job-state.

For production scale this should be replaced with Celery, but the
contract is the same and the API caller never needs to know the
difference.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from latentspec.db import session_scope
from latentspec.models import JobKind, JobStatus, MiningJob

log = logging.getLogger(__name__)


class JobContext:
    """Passed to handlers; provides progress reporting + cancel signal."""

    def __init__(self, job_id: uuid.UUID, agent_id: uuid.UUID) -> None:
        self.job_id = job_id
        self.agent_id = agent_id
        self._cancel = asyncio.Event()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    async def report_progress(self, percent: float, message: str | None = None) -> None:
        async with session_scope() as session:
            row = await session.get(MiningJob, self.job_id)
            if row is None:
                return
            row.progress_percent = max(0.0, min(100.0, float(percent)))
            if message is not None:
                row.progress_message = message[:512]


class JobHandler(Protocol):
    async def __call__(self, ctx: JobContext, config: dict[str, Any]) -> dict[str, Any]: ...


class JobRunner(ABC):
    @abstractmethod
    async def submit(
        self, *, agent_id: uuid.UUID, kind: JobKind, config: dict[str, Any]
    ) -> uuid.UUID: ...

    @abstractmethod
    async def cancel(self, job_id: uuid.UUID) -> bool: ...


# ---- in-process implementation ------------------------------------------


class InProcessJobRunner(JobRunner):
    def __init__(self, *, max_concurrency: int = 4) -> None:
        self._handlers: dict[JobKind, JobHandler] = {}
        self._sem = asyncio.Semaphore(max_concurrency)
        self._tasks: dict[uuid.UUID, tuple[asyncio.Task, JobContext]] = {}

    def register_handler(self, kind: JobKind, handler: JobHandler) -> None:
        self._handlers[kind] = handler

    async def submit(
        self, *, agent_id: uuid.UUID, kind: JobKind, config: dict[str, Any]
    ) -> uuid.UUID:
        async with session_scope() as session:
            job = MiningJob(
                agent_id=agent_id,
                kind=kind,
                status=JobStatus.PENDING,
                config=config or {},
            )
            session.add(job)
            await session.flush()
            await session.refresh(job)
            job_id = job.id

        ctx = JobContext(job_id=job_id, agent_id=agent_id)
        task = asyncio.create_task(self._run(job_id, kind, config, ctx))
        self._tasks[job_id] = (task, ctx)
        return job_id

    async def cancel(self, job_id: uuid.UUID) -> bool:
        entry = self._tasks.get(job_id)
        if entry is None:
            return False
        task, ctx = entry
        ctx.cancel()
        if not task.done():
            task.cancel()
        async with session_scope() as session:
            row = await session.get(MiningJob, job_id)
            if row is not None and row.status in (JobStatus.PENDING, JobStatus.RUNNING):
                row.status = JobStatus.CANCELLED
                row.completed_at = datetime.now(UTC)
        return True

    async def _run(
        self,
        job_id: uuid.UUID,
        kind: JobKind,
        config: dict[str, Any],
        ctx: JobContext,
    ) -> None:
        handler = self._handlers.get(kind)
        if handler is None:
            await self._fail(job_id, f"no handler registered for {kind.value}")
            return

        async with self._sem:
            async with session_scope() as session:
                row = await session.get(MiningJob, job_id)
                if row is None or row.status == JobStatus.CANCELLED:
                    return
                row.status = JobStatus.RUNNING
                row.started_at = datetime.now(UTC)

            try:
                result = await handler(ctx, config)
            except asyncio.CancelledError:
                async with session_scope() as session:
                    row = await session.get(MiningJob, job_id)
                    if row is not None:
                        row.status = JobStatus.CANCELLED
                        row.completed_at = datetime.now(UTC)
                return
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc()
                log.exception("job %s failed", job_id)
                await self._fail(job_id, f"{e!r}\n{tb[-2000:]}")
                return

            async with session_scope() as session:
                row = await session.get(MiningJob, job_id)
                if row is None:
                    return
                row.status = JobStatus.SUCCEEDED
                row.completed_at = datetime.now(UTC)
                row.progress_percent = 100.0
                row.result = result

    async def _fail(self, job_id: uuid.UUID, error: str) -> None:
        async with session_scope() as session:
            row = await session.get(MiningJob, job_id)
            if row is None:
                return
            row.status = JobStatus.FAILED
            row.completed_at = datetime.now(UTC)
            row.error = error[:8000]


_singleton: InProcessJobRunner | None = None


def get_runner() -> InProcessJobRunner:
    global _singleton
    if _singleton is None:
        _singleton = InProcessJobRunner()
    return _singleton


def register_handler(kind: JobKind, handler: JobHandler) -> None:
    get_runner().register_handler(kind, handler)


async def enqueue_job(
    *, agent_id: uuid.UUID, kind: JobKind, config: dict[str, Any]
) -> uuid.UUID:
    return await get_runner().submit(agent_id=agent_id, kind=kind, config=config)
