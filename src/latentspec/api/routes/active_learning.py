"""Active-learning API — synthetic generation + persisted review queue."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from latentspec.active_learning.synthesis import (
    AgentSpec,
    deterministic_synthetic_traces,
    generate_synthetic_traces,
)
from latentspec.db import get_db
from latentspec.models import Agent, ReviewDecision, SyntheticReviewItem
from latentspec.schemas.trace import NormalizedTrace

router = APIRouter()


class SynthesizeIn(BaseModel):
    spec: dict[str, Any]
    n_traces: int = Field(default=10, ge=1, le=50)
    use_llm: bool = True


class SynthesizedItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    agent_id: uuid.UUID
    spec_name: str
    decision: ReviewDecision
    decided_at: datetime | None
    decided_by: str | None
    edit_notes: str | None
    created_at: datetime
    trace: dict[str, Any]

    @classmethod
    def from_row(cls, row: SyntheticReviewItem) -> "SynthesizedItemOut":
        return cls(
            id=row.id,
            agent_id=row.agent_id,
            spec_name=row.spec_name,
            decision=row.decision,
            decided_at=row.decided_at,
            decided_by=row.decided_by,
            edit_notes=row.edit_notes,
            created_at=row.created_at,
            trace=row.trace_data,
        )


@router.post(
    "/agents/{agent_id}/active-learning/synthesize",
    response_model=list[SynthesizedItemOut],
    status_code=201,
)
async def synthesize_traces(
    agent_id: uuid.UUID,
    payload: SynthesizeIn,
    db: AsyncSession = Depends(get_db),
) -> list[SynthesizedItemOut]:
    agent = await db.get(Agent, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="agent not found")

    spec = AgentSpec(
        name=payload.spec.get("name", agent.name),
        purpose=payload.spec.get("purpose", agent.description or ""),
        tools=list(payload.spec.get("tools") or []),
        sample_user_inputs=list(payload.spec.get("sample_user_inputs") or []),
        user_segments=list(payload.spec.get("user_segments") or []),
        forbidden_actions=list(payload.spec.get("forbidden_actions") or []),
        typical_session_length=tuple(
            payload.spec.get("typical_session_length") or (3, 8)
        ),  # type: ignore[arg-type]
    )

    traces: list[NormalizedTrace]
    if payload.use_llm:
        try:
            traces = await generate_synthetic_traces(spec, n_traces=payload.n_traces)
        except Exception:
            traces = []
    else:
        traces = []

    if not traces:
        traces = deterministic_synthetic_traces(spec, n_traces=payload.n_traces)

    out: list[SynthesizedItemOut] = []
    for t in traces:
        row = SyntheticReviewItem(
            agent_id=agent_id,
            spec_name=spec.name,
            trace_data=t.model_dump(mode="json"),
            decision=ReviewDecision.PENDING,
        )
        db.add(row)
        await db.flush()
        await db.refresh(row)
        out.append(SynthesizedItemOut.from_row(row))
    return out


@router.get(
    "/agents/{agent_id}/active-learning/queue",
    response_model=list[SynthesizedItemOut],
)
async def list_queue(
    agent_id: uuid.UUID,
    decision: ReviewDecision | None = None,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
) -> list[SynthesizedItemOut]:
    stmt = (
        select(SyntheticReviewItem)
        .where(SyntheticReviewItem.agent_id == agent_id)
        .order_by(SyntheticReviewItem.created_at.desc())
        .limit(limit)
    )
    if decision is not None:
        stmt = stmt.where(SyntheticReviewItem.decision == decision)
    rows = list((await db.execute(stmt)).scalars().all())
    return [SynthesizedItemOut.from_row(r) for r in rows]


class DecisionIn(BaseModel):
    decision: ReviewDecision
    decided_by: str | None = None
    edit_notes: str | None = None
    replacement_trace: dict[str, Any] | None = None


@router.patch(
    "/active-learning/queue/{item_id}",
    response_model=SynthesizedItemOut,
)
async def decide_item(
    item_id: uuid.UUID,
    payload: DecisionIn,
    db: AsyncSession = Depends(get_db),
) -> SynthesizedItemOut:
    row = await db.get(SyntheticReviewItem, item_id)
    if row is None:
        raise HTTPException(status_code=404, detail="queue item not found")
    row.decision = payload.decision
    row.decided_at = datetime.now(UTC)
    row.decided_by = payload.decided_by
    row.edit_notes = payload.edit_notes
    if payload.replacement_trace and payload.decision == ReviewDecision.EDITED:
        # Validate the replacement against the §3.2 schema before persisting
        try:
            NormalizedTrace.model_validate(payload.replacement_trace)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"replacement trace invalid: {e}"
            ) from e
        row.trace_data = payload.replacement_trace
    await db.flush()
    await db.refresh(row)
    return SynthesizedItemOut.from_row(row)
