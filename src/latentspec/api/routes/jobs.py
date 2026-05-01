"""Background job API — submit / poll / cancel.

Used by every async-capable orchestrator: mining, cluster mining,
calibration, synthetic generation, pack autofit. The handlers themselves
live in `latentspec.jobs.handlers`.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from latentspec.db import get_db
from latentspec.jobs.handlers import register_default_handlers
from latentspec.jobs.runner import enqueue_job, get_runner
from latentspec.models import JobKind, JobStatus, MiningJob

router = APIRouter()

# Wire handlers on import — idempotent.
register_default_handlers()


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    agent_id: uuid.UUID
    kind: JobKind
    status: JobStatus
    submitted_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    progress_percent: float
    progress_message: str | None
    config: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None


class JobSubmitIn(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)


@router.post("/agents/{agent_id}/jobs/{kind}", response_model=JobOut, status_code=202)
async def submit_job(
    agent_id: uuid.UUID,
    kind: JobKind,
    payload: JobSubmitIn,
    db: AsyncSession = Depends(get_db),
) -> MiningJob:
    job_id = await enqueue_job(agent_id=agent_id, kind=kind, config=payload.config)
    job = await db.get(MiningJob, job_id)
    if job is None:
        raise HTTPException(status_code=500, detail="failed to persist job")
    return job


@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> MiningJob:
    job = await db.get(MiningJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.get("/agents/{agent_id}/jobs", response_model=list[JobOut])
async def list_jobs(
    agent_id: uuid.UUID,
    status: JobStatus | None = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
) -> list[MiningJob]:
    stmt = (
        select(MiningJob)
        .where(MiningJob.agent_id == agent_id)
        .order_by(MiningJob.submitted_at.desc())
        .limit(limit)
    )
    if status is not None:
        stmt = stmt.where(MiningJob.status == status)
    return list((await db.execute(stmt)).scalars().all())


@router.delete("/jobs/{job_id}", status_code=204)
async def cancel_job(job_id: uuid.UUID) -> None:
    ok = await get_runner().cancel(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="job not found / already done")
