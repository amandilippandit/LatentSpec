"""Session API — multi-turn session ingestion + retrieval + mining.

A session groups N traces (turns) under one `session_id`. The actual
turn payloads are still ingested via `POST /traces` — this router only
manages the session metadata + provides session-level mining triggers.
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
from latentspec.models import Session as SessionModel
from latentspec.models import Trace
from latentspec.schemas.trace import NormalizedTrace
from latentspec.sessions.orchestrator import mine_session_invariants
from latentspec.sessions.schema import Session as SessionSchema

router = APIRouter()


class SessionIn(BaseModel):
    agent_id: uuid.UUID
    external_id: str | None = Field(default=None, max_length=128)
    user_id: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    agent_id: uuid.UUID
    external_id: str | None
    user_id: str | None
    started_at: datetime
    ended_at: datetime | None
    n_turns: int


@router.post("", response_model=SessionOut, status_code=201)
async def create_session(payload: SessionIn, db: AsyncSession = Depends(get_db)) -> SessionModel:
    row = SessionModel(
        agent_id=payload.agent_id,
        external_id=payload.external_id,
        user_id=payload.user_id,
        metadata_=payload.metadata,
    )
    db.add(row)
    await db.flush()
    await db.refresh(row)
    return row


@router.post("/{session_id}/close", response_model=SessionOut)
async def close_session(
    session_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> SessionModel:
    row = await db.get(SessionModel, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="session not found")
    row.ended_at = datetime.utcnow()
    await db.flush()
    return row


@router.get("/{session_id}", response_model=SessionOut)
async def get_session(
    session_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> SessionModel:
    row = await db.get(SessionModel, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="session not found")
    return row


class SessionMiningOut(BaseModel):
    n_sessions: int
    n_transitions: int
    n_aggregates: int
    n_terminations: int
    sample_invariants: list[dict[str, Any]]


@router.post("/agents/{agent_id}/session-mining", response_model=SessionMiningOut)
async def trigger_session_mining(
    agent_id: uuid.UUID,
    limit_sessions: int = 200,
    db: AsyncSession = Depends(get_db),
) -> SessionMiningOut:
    """Build sessions from `Session` rows + correlated `Trace` rows, then mine."""
    session_rows = (
        await db.execute(
            select(SessionModel)
            .where(SessionModel.agent_id == agent_id)
            .order_by(SessionModel.started_at.desc())
            .limit(limit_sessions)
        )
    ).scalars().all()

    sessions: list[SessionSchema] = []
    for s_row in session_rows:
        # Find traces tagged with this session's id (string form)
        traces = (
            await db.execute(
                select(Trace)
                .where(Trace.agent_id == agent_id)
                .where(Trace.session_id == str(s_row.id))
                .order_by(Trace.started_at.asc())
            )
        ).scalars().all()
        if not traces:
            continue
        sessions.append(
            SessionSchema(
                session_id=str(s_row.id),
                agent_id=str(agent_id),
                user_id=s_row.user_id,
                started_at=s_row.started_at,
                ended_at=s_row.ended_at,
                turns=[NormalizedTrace.model_validate(t.trace_data) for t in traces],
            )
        )

    if not sessions:
        return SessionMiningOut(
            n_sessions=0, n_transitions=0, n_aggregates=0, n_terminations=0,
            sample_invariants=[],
        )

    result = mine_session_invariants(sessions)

    samples = []
    for c in (result.transitions + result.aggregates + result.terminations)[:10]:
        samples.append(
            {
                "type": c.type.value,
                "description": c.description,
                "support": c.support,
                "consistency": c.consistency,
                "session_level": True,
            }
        )

    return SessionMiningOut(
        n_sessions=result.n_sessions,
        n_transitions=len(result.transitions),
        n_aggregates=len(result.aggregates),
        n_terminations=len(result.terminations),
        sample_invariants=samples,
    )
