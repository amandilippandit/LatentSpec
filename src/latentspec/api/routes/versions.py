"""Agent-version management API.

  GET    /agents/{id}/versions                     — list known versions
  GET    /agents/{id}/versions/{tag}               — single version detail
  GET    /agents/{id}/versions/{a}/diff/{b}        — VersionDelta(a, b)
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from latentspec.db import get_db
from latentspec.models import AgentVersion
from latentspec.versioning.tracker import diff_versions

router = APIRouter()


class VersionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    version_tag: str
    tool_repertoire: list[str]
    first_seen_at: datetime
    last_seen_at: datetime
    parent_version_tag: str | None


@router.get("/agents/{agent_id}/versions", response_model=list[VersionOut])
async def list_versions(
    agent_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> list[AgentVersion]:
    return list(
        (
            await db.execute(
                select(AgentVersion)
                .where(AgentVersion.agent_id == agent_id)
                .order_by(AgentVersion.last_seen_at.desc())
            )
        ).scalars().all()
    )


@router.get("/agents/{agent_id}/versions/{tag}", response_model=VersionOut)
async def get_version(
    agent_id: uuid.UUID, tag: str, db: AsyncSession = Depends(get_db)
) -> AgentVersion:
    row = (
        await db.execute(
            select(AgentVersion)
            .where(AgentVersion.agent_id == agent_id)
            .where(AgentVersion.version_tag == tag)
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="version not found")
    return row


class VersionDiffOut(BaseModel):
    added: list[str]
    removed: list[str]
    common: list[str]
    likely_renames: list[list[str]]
    is_breaking: bool


@router.get(
    "/agents/{agent_id}/versions/{a}/diff/{b}", response_model=VersionDiffOut
)
async def diff_versions_endpoint(
    agent_id: uuid.UUID, a: str, b: str, db: AsyncSession = Depends(get_db)
) -> VersionDiffOut:
    rows = list(
        (
            await db.execute(
                select(AgentVersion)
                .where(AgentVersion.agent_id == agent_id)
                .where(AgentVersion.version_tag.in_([a, b]))
            )
        ).scalars().all()
    )
    by_tag = {row.version_tag: row for row in rows}
    if a not in by_tag or b not in by_tag:
        raise HTTPException(status_code=404, detail="one or both versions not found")
    delta = diff_versions(by_tag[a].tool_repertoire or [], by_tag[b].tool_repertoire or [])
    return VersionDiffOut(
        added=delta.added,
        removed=delta.removed,
        common=delta.common,
        likely_renames=[[old, new] for old, new in delta.likely_renames],
        is_breaking=delta.is_breaking,
    )
