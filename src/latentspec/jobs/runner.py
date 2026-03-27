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
