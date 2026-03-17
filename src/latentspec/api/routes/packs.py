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
            status=InvariantStatus.PENDING,
            evidence_trace_ids=[],
            evidence_count=0,
            violation_count=0,
            discovered_by="pack",
            params=inv.params,
        )
        db.add(row)
        await db.flush()
        invariant_ids.append(row.id)

    pack_obj = get_pack(pack_id)
    return InstallPackOut(
        n_installed=len(invariant_ids),
        pack_id=pack_id,
        pack_version=pack_obj.version if pack_obj else "0.0.0",
        invariant_ids=invariant_ids,
    )


class AutoFitOut(BaseModel):
    invariant_id: uuid.UUID
    fit: float
    applicability: float
    pass_rate: float
    sample_size: int
    suggested_status: str


@router.post(
    "/agents/{agent_id}/packs/auto-fit",
    response_model=list[AutoFitOut],
)
async def auto_fit_pack_invariants(
    agent_id: uuid.UUID,
    sample_limit: int = 200,
    db: AsyncSession = Depends(get_db),
) -> list[AutoFitOut]:
    """Score each pack-installed pending invariant against recent traces.

    Returns suggested status transitions. Caller decides whether to
    apply them by PATCHing each invariant.
    """
    rows = (
        await db.execute(
            select(Invariant)
            .where(Invariant.agent_id == agent_id)
            .where(Invariant.discovered_by == "pack")
        )
    ).scalars().all()

    trace_rows = (
        await db.execute(
            select(Trace)
            .where(Trace.agent_id == agent_id)
            .order_by(Trace.started_at.desc())
            .limit(sample_limit)
        )
    ).scalars().all()
    sample = [NormalizedTrace.model_validate(r.trace_data) for r in trace_rows]

    out: list[AutoFitOut] = []
    for inv in rows:
        from latentspec.schemas.invariant import MinedInvariant

        mined = MinedInvariant(
            invariant_id=str(inv.id),
            type=inv.type,
            description=inv.description,
            formal_rule=inv.formal_rule,
            confidence=inv.confidence,
            support_score=inv.support_score,
            consistency_score=inv.consistency_score,
            cross_val_bonus=inv.cross_val_bonus,
            clarity_score=inv.clarity_score,
            evidence_count=inv.evidence_count,
            violation_count=inv.violation_count,
            discovered_at=inv.discovered_at,
            status=inv.status,
            severity=inv.severity,
            discovered_by="pack",
            evidence_trace_ids=[],
            params=dict(inv.params or {}),
        )
        score = auto_fit_score(invariant=mined, traces=sample)
        if score.fit >= 0.85:
            suggested = "active"
        elif score.fit >= 0.6:
            suggested = "pending"
        else:
            suggested = "rejected"
        out.append(
            AutoFitOut(
                invariant_id=inv.id,
                fit=score.fit,
                applicability=score.applicability,
                pass_rate=score.pass_rate,
                sample_size=score.sample_size,
                suggested_status=suggested,
            )
        )

    return out
