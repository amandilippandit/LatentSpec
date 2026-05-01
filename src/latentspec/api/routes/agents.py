"""Agent registration + listing routes."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from latentspec.db import get_db
from latentspec.models import Agent

router = APIRouter()


class AgentIn(BaseModel):
    org_id: uuid.UUID
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    framework: str | None = Field(default=None, max_length=64)


class AgentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    org_id: uuid.UUID
    name: str
    description: str | None
    framework: str | None
    created_at: datetime


@router.post("", response_model=AgentOut, status_code=status.HTTP_201_CREATED)
async def create_agent(payload: AgentIn, db: AsyncSession = Depends(get_db)) -> Agent:
    agent = Agent(**payload.model_dump())
    db.add(agent)
    await db.flush()
    await db.refresh(agent)
    return agent


@router.get("", response_model=list[AgentOut])
async def list_agents(
    org_id: uuid.UUID | None = None,
    db: AsyncSession = Depends(get_db),
) -> list[Agent]:
    stmt = select(Agent).order_by(Agent.created_at.desc())
    if org_id is not None:
        stmt = stmt.where(Agent.org_id == org_id)
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.get("/{agent_id}", response_model=AgentOut)
async def get_agent(agent_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> Agent:
    agent = await db.get(Agent, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return agent
