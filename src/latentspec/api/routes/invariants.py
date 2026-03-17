"""Invariant management routes — list, fetch, confirm/reject/edit (§6.1).

Implements the Invariant Explorer surface from the dashboard spec:
  GET    /invariants?agent_id=&status=&type=&min_confidence=
  GET    /invariants/{id}
  PATCH  /invariants/{id}                  — confirm / reject / edit
"""

from __future__ import annotations

import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from latentspec.db import get_db
from latentspec.models import (
    Invariant,
    InvariantStatus,
    InvariantType,
    Severity,
)
from latentspec.schemas.invariant import InvariantOut

router = APIRouter()


@router.get("", response_model=list[InvariantOut])
async def list_invariants(
    agent_id: uuid.UUID,
    status: InvariantStatus | None = None,
    type: InvariantType | None = None,
    min_confidence: float = 0.0,
    db: AsyncSession = Depends(get_db),
) -> list[Invariant]:
    stmt = (
        select(Invariant)
        .where(Invariant.agent_id == agent_id)
        .where(Invariant.confidence >= min_confidence)
        .order_by(Invariant.confidence.desc())
    )
    if status is not None:
        stmt = stmt.where(Invariant.status == status)
    if type is not None:
        stmt = stmt.where(Invariant.type == type)
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.get("/{invariant_id}", response_model=InvariantOut)
async def get_invariant(
    invariant_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> Invariant:
    inv = await db.get(Invariant, invariant_id)
    if inv is None:
        raise HTTPException(status_code=404, detail="invariant not found")
    return inv


class InvariantPatch(BaseModel):
    """One-click confirm/reject/edit (§6.1 Invariant Explorer)."""

    action: Literal["confirm", "reject", "edit"] | None = None
    description: str | None = None
    severity: Severity | None = None


@router.patch("/{invariant_id}", response_model=InvariantOut)
async def patch_invariant(
    invariant_id: uuid.UUID,
    payload: InvariantPatch,
    db: AsyncSession = Depends(get_db),
) -> Invariant:
    inv = await db.get(Invariant, invariant_id)
    if inv is None:
        raise HTTPException(status_code=404, detail="invariant not found")

    if payload.action == "confirm":
        inv.status = InvariantStatus.ACTIVE
    elif payload.action == "reject":
        inv.status = InvariantStatus.REJECTED
    if payload.description is not None:
        inv.description = payload.description.strip()
    if payload.severity is not None:
        inv.severity = payload.severity
    await db.flush()
    await db.refresh(inv)
    return inv
