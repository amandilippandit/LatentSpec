"""Mining trigger route — `POST /agents/{id}/mining-runs`."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from latentspec.db import get_db
from latentspec.mining.orchestrator import mine_invariants
from latentspec.models import Agent, Trace
from latentspec.schemas.invariant import MinedInvariant
from latentspec.schemas.trace import NormalizedTrace

router = APIRouter()


class MiningTriggerIn(BaseModel):
    """Optional knobs that override the mining_runs.config defaults."""

    limit: int = Field(default=500, ge=1, le=5000)
    version_tag: str | None = None


class MiningTriggerOut(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    mining_run_id: uuid.UUID | None
    agent_id: uuid.UUID
    traces_analyzed: int
    candidates_statistical: int
    candidates_llm: int
    candidates_total_unique: int
    by_status: dict[str, int]
    by_type: dict[str, int]
    duration_seconds: float
    invariants: list[MinedInvariant]
    started_at: datetime


@router.post("/{agent_id}/mining-runs", response_model=MiningTriggerOut)
async def trigger_mining(
    agent_id: uuid.UUID,
    payload: MiningTriggerIn,
    db: AsyncSession = Depends(get_db),
) -> MiningTriggerOut:
    agent = await db.get(Agent, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="agent not found")

    stmt = (
        select(Trace)
        .where(Trace.agent_id == agent_id)
        .order_by(Trace.started_at.desc())
        .limit(payload.limit)
    )
    if payload.version_tag is not None:
        stmt = stmt.where(Trace.version_tag == payload.version_tag)

    result = await db.execute(stmt)
    rows = list(result.scalars().all())
    traces = [NormalizedTrace.model_validate(row.trace_data) for row in rows]

    started = datetime.now()
    mining_result = await mine_invariants(
        agent_id=agent_id,
        traces=traces,
        session=db,
        persist=True,
    )

    return MiningTriggerOut(
        mining_run_id=mining_result.mining_run_id,
        agent_id=mining_result.agent_id,
        traces_analyzed=mining_result.traces_analyzed,
        candidates_statistical=mining_result.candidates_statistical,
        candidates_llm=mining_result.candidates_llm,
        candidates_total_unique=mining_result.candidates_total_unique,
        by_status=mining_result.by_status,
        by_type=mining_result.by_type,
        duration_seconds=mining_result.duration_seconds,
        invariants=mining_result.invariants,
        started_at=started,
    )
