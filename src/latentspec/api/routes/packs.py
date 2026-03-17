"""Vertical pack management API."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from latentspec.db import get_db
from latentspec.models import (
    Agent,
    Invariant,
    InvariantStatus,
    InvariantType,
    Severity,
    Trace,
)
from latentspec.packs import (
    auto_fit_score,
    get_pack,
    install_pack,
    list_packs,
)
from latentspec.schemas.trace import NormalizedTrace

router = APIRouter()


class PackOut(BaseModel):
    pack_id: str
    title: str
    description: str
    version: str
    n_invariants: int
    regulatory_frameworks: list[str]
    targets: list[str]


@router.get("/packs", response_model=list[PackOut])
async def list_available_packs() -> list[PackOut]:
    out: list[PackOut] = []
    for pid in list_packs():
        pack = get_pack(pid)
        if pack is None:
            continue
        out.append(
            PackOut(
                pack_id=pack.pack_id,
                title=pack.title,
                description=pack.description,
                version=pack.version,
                n_invariants=len(pack.invariants),
                regulatory_frameworks=pack.regulatory_frameworks,
                targets=pack.targets,
            )
        )
    return out


class InstallPackOut(BaseModel):
    n_installed: int
    pack_id: str
    pack_version: str
    invariant_ids: list[uuid.UUID]


@router.post(
    "/agents/{agent_id}/packs/{pack_id}/install",
    response_model=InstallPackOut,
    status_code=201,
)
async def install_pack_into_agent(
    agent_id: uuid.UUID,
    pack_id: str,
    db: AsyncSession = Depends(get_db),
) -> InstallPackOut:
    agent = await db.get(Agent, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="agent not found")

    try:
        materialized = install_pack(agent_id=agent_id, pack_id=pack_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    invariant_ids: list[uuid.UUID] = []
    for inv in materialized:
        row = Invariant(
            agent_id=agent_id,
            type=InvariantType(inv.type),
            description=inv.description,
            formal_rule=inv.formal_rule,
            confidence=inv.confidence,
            support_score=inv.support_score,
            consistency_score=inv.consistency_score,
            cross_val_bonus=inv.cross_val_bonus,
            clarity_score=inv.clarity_score,
            severity=Severity(inv.severity),
