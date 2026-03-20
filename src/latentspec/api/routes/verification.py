"""Z3 verification routes (§3.2 / §10.1).

  POST /invariants/{id}/verify       — run Z3 verification against a sample
  POST /invariants/{id}/certificate  — generate a §10.1 verification certificate

Both routes use the same SMT engine; certificates additionally carry a
cryptographic signature when `LATENTSPEC_CERT_SIGNING_KEY` is set.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from latentspec.db import get_db
from latentspec.models import Invariant, InvariantType, Trace
from latentspec.schemas.trace import NormalizedTrace
from latentspec.smt.certificates import generate_certificate
from latentspec.smt.compiler import compile_invariant
from latentspec.smt.verifier import verify_trace

router = APIRouter()


class VerifyIn(BaseModel):
    sample_size: int = Field(default=20, ge=1, le=500)
    timeout_ms_per_trace: int = Field(default=100, ge=10, le=5000)
    version_tag: str | None = None


class VerifyOut(BaseModel):
    invariant_id: str
    sample_size: int
    holds: int
    violates: int
    pass_rate: float
    counter_examples: list[dict]


@router.post("/{invariant_id}/verify", response_model=VerifyOut)
async def run_verification(
    invariant_id: uuid.UUID,
    payload: VerifyIn,
    db: AsyncSession = Depends(get_db),
) -> VerifyOut:
    inv = await db.get(Invariant, invariant_id)
    if inv is None:
        raise HTTPException(status_code=404, detail="invariant not found")

    if inv.type == InvariantType.OUTPUT_FORMAT:
        raise HTTPException(
            status_code=400,
            detail="output_format invariants use LLM-as-judge, not Z3",
        )

    try:
        compilation = compile_invariant(inv.type, dict(inv.params or {}))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"compile failed: {e}") from e

    stmt = (
        select(Trace)
        .where(Trace.agent_id == inv.agent_id)
        .order_by(Trace.started_at.desc())
        .limit(payload.sample_size)
    )
    if payload.version_tag is not None:
        stmt = stmt.where(Trace.version_tag == payload.version_tag)
    rows = list((await db.execute(stmt)).scalars().all())
    if not rows:
        raise HTTPException(status_code=400, detail="no traces available for this agent")

    holds = 0
    violates = 0
    counters: list[dict] = []
    for row in rows:
        nt = NormalizedTrace.model_validate(row.trace_data)
        result = verify_trace(compilation, nt, timeout_ms=payload.timeout_ms_per_trace)
        if result.holds:
            holds += 1
        else:
            violates += 1
            if result.counter_example is not None and len(counters) < 3:
                counters.append(
                    {"trace_id": nt.trace_id, "counter_example": result.counter_example}
                )

    n = holds + violates
    return VerifyOut(
        invariant_id=str(invariant_id),
        sample_size=n,
        holds=holds,
        violates=violates,
        pass_rate=round(holds / max(1, n), 4),
        counter_examples=counters,
    )


@router.post("/{invariant_id}/certificate")
async def issue_certificate(
    invariant_id: uuid.UUID,
    payload: VerifyIn,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """§10.1 — premium enterprise certificate. Hidden surface today."""
    inv = await db.get(Invariant, invariant_id)
    if inv is None:
        raise HTTPException(status_code=404, detail="invariant not found")
    try:
        compilation = compile_invariant(inv.type, dict(inv.params or {}))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"compile failed: {e}") from e

    rows = list(
        (
            await db.execute(
                select(Trace)
                .where(Trace.agent_id == inv.agent_id)
                .order_by(Trace.started_at.desc())
                .limit(payload.sample_size)
            )
        )
        .scalars()
        .all()
    )
    sample = [NormalizedTrace.model_validate(r.trace_data) for r in rows]
    if not sample:
        raise HTTPException(status_code=400, detail="no traces available")

    cert = generate_certificate(
        compilation,
        sample,
        timeout_ms_per_trace=payload.timeout_ms_per_trace,
    )
    return asdict(cert)
