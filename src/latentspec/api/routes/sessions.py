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
