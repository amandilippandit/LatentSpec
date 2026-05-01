"""Fingerprint baseline API — read-only view of an agent's distribution."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from latentspec.db import get_db
from latentspec.models import FingerprintBaseline

router = APIRouter()


class FingerprintBaselineOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    agent_id: uuid.UUID
    n_observed: int
    n_distinct_baseline: int
    n_distinct_observed: int
    chi_square_threshold: float
    last_drift_at: datetime | None
    last_drift_chi: float | None
    drifting: bool


@router.get(
    "/agents/{agent_id}/fingerprints/baseline",
    response_model=FingerprintBaselineOut,
)
async def get_baseline(
    agent_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> FingerprintBaselineOut:
    row = (
        await db.execute(
            select(FingerprintBaseline).where(FingerprintBaseline.agent_id == agent_id)
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="no baseline yet — run mining first")
    return FingerprintBaselineOut(
        agent_id=row.agent_id,
        n_observed=row.n_observed,
        n_distinct_baseline=len(row.baseline_counts or {}),
        n_distinct_observed=len(row.observed_counts or {}),
        chi_square_threshold=row.chi_square_threshold,
        last_drift_at=row.last_drift_at,
        last_drift_chi=row.last_drift_chi,
        drifting=row.last_drift_at is not None
        and (
            row.last_drift_chi is not None
            and row.last_drift_chi > row.chi_square_threshold
        ),
    )


class FingerprintTopOut(BaseModel):
    fingerprint: str
    baseline_count: int
    observed_count: int


@router.get(
    "/agents/{agent_id}/fingerprints/top",
    response_model=list[FingerprintTopOut],
)
async def get_top_fingerprints(
    agent_id: uuid.UUID,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
) -> list[FingerprintTopOut]:
    row = (
        await db.execute(
            select(FingerprintBaseline).where(FingerprintBaseline.agent_id == agent_id)
        )
    ).scalar_one_or_none()
    if row is None:
        return []
    keys = sorted(
        set((row.baseline_counts or {}).keys()) | set((row.observed_counts or {}).keys()),
        key=lambda k: -((row.observed_counts or {}).get(k, 0)),
    )[:limit]
    return [
        FingerprintTopOut(
            fingerprint=k,
            baseline_count=(row.baseline_counts or {}).get(k, 0),
            observed_count=(row.observed_counts or {}).get(k, 0),
        )
        for k in keys
    ]
