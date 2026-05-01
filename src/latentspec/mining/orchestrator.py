"""Mining orchestrator — runs the full Stage 1→2→3 pipeline on a trace batch.

Executes Track A and Track B *in parallel*, cross-validates, scores, and
formalizes. Returns a structured `MiningResult` and (optionally) persists
the run to the `mining_runs` and `invariants` tables.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from latentspec.config import get_settings
from latentspec.mining.confidence import cross_validate
from latentspec.mining.formalization import formalize
from latentspec.mining.llm.runner import run_llm_track
from latentspec.mining.statistical.runner import run_statistical_track
from latentspec.models.invariant import (
    Invariant,
    InvariantStatus,
    InvariantType,
    Severity,
)
from latentspec.models.mining_run import MiningRun, MiningRunStatus
from latentspec.schemas.invariant import InvariantCandidate, MinedInvariant
from latentspec.schemas.trace import NormalizedTrace

log = logging.getLogger(__name__)


@dataclass
class MiningResult:
    agent_id: uuid.UUID
    mining_run_id: uuid.UUID | None
    traces_analyzed: int
    candidates_statistical: int
    candidates_llm: int
    candidates_total_unique: int
    invariants: list[MinedInvariant] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def by_status(self) -> dict[str, int]:
        out = {s.value: 0 for s in InvariantStatus}
        for inv in self.invariants:
            out[inv.status.value] += 1
        return out

    @property
    def by_type(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for inv in self.invariants:
            out[inv.type.value] = out.get(inv.type.value, 0) + 1
        return out


async def mine_invariants(
    *,
    agent_id: uuid.UUID,
    traces: list[NormalizedTrace],
    session: AsyncSession | None = None,
    persist: bool = True,
    config_overrides: dict[str, Any] | None = None,
) -> MiningResult:
    """Run the full §3.2 mining pipeline on a batch of normalized traces.

    Args:
        agent_id: agent these traces belong to.
        traces: normalized §3.2 traces (Stage 1 output).
        session: an open async DB session if persist=True.
        persist: when True, write the MiningRun row and persist new
            invariants. When False, just return the in-memory result
            (used for local demos and tests).
        config_overrides: arbitrary extra config to record on the run row.
    """
    settings = get_settings()
    started = datetime.now(UTC)
    started_perf = asyncio.get_event_loop().time()

    config = {
        "mining_batch_size": settings.mining_batch_size,
        "mining_min_support": settings.mining_min_support,
        "confidence_reject_threshold": settings.confidence_reject_threshold,
        "confidence_review_threshold": settings.confidence_review_threshold,
        **(config_overrides or {}),
    }

    mining_run_id: uuid.UUID | None = None
    if persist and session is not None:
        run = MiningRun(
            agent_id=agent_id,
            started_at=started,
            traces_analyzed=len(traces),
            config=config,
            status=MiningRunStatus.RUNNING,
        )
        session.add(run)
        await session.flush()
        mining_run_id = run.id

    try:
        statistical_task = asyncio.create_task(
            asyncio.to_thread(
                run_statistical_track,
                traces,
                min_support_sequence=settings.mining_min_support,
            )
        )
        llm_task = asyncio.create_task(run_llm_track(traces))

        statistical: list[InvariantCandidate]
        llm: list[InvariantCandidate]
        statistical, llm = await asyncio.gather(statistical_task, llm_task)

        merged = cross_validate(statistical, llm)

        invariants: list[MinedInvariant] = []
        dropped_invalid_params = 0
        for cand in merged:
            inv = formalize(
                cand,
                reject_threshold=settings.confidence_reject_threshold,
                review_threshold=settings.confidence_review_threshold,
            )
            if inv is None:
                # Failed params validation — drop rather than persist a
                # rule the runtime checker can't evaluate.
                dropped_invalid_params += 1
                continue
            if inv.status == InvariantStatus.REJECTED:
                continue
            invariants.append(inv)

        invariants.sort(key=lambda i: -i.confidence)
        if dropped_invalid_params:
            log.info(
                "dropped %d candidates with invalid params during mining run",
                dropped_invalid_params,
            )

        if persist and session is not None:
            await _persist_invariants(session, agent_id, invariants)

        duration = asyncio.get_event_loop().time() - started_perf

        if persist and session is not None and mining_run_id is not None:
            run.completed_at = datetime.now(UTC)
            run.invariants_discovered = len(invariants)
            run.status = MiningRunStatus.COMPLETED
            await session.flush()

        return MiningResult(
            agent_id=agent_id,
            mining_run_id=mining_run_id,
            traces_analyzed=len(traces),
            candidates_statistical=len(statistical),
            candidates_llm=len(llm),
            candidates_total_unique=len(merged),
            invariants=invariants,
            duration_seconds=round(duration, 3),
        )

    except Exception as e:
        log.exception("mining run failed")
        if persist and session is not None and mining_run_id is not None:
            run.status = MiningRunStatus.FAILED
            run.error = repr(e)[:1000]
            run.completed_at = datetime.now(UTC)
            await session.flush()
        raise


async def _persist_invariants(
    session: AsyncSession,
    agent_id: uuid.UUID,
    invariants: list[MinedInvariant],
) -> None:
    """Insert new invariants; update existing ones by matching (agent_id, type, description)."""
    if not invariants:
        return

    existing = await session.execute(
        select(Invariant).where(Invariant.agent_id == agent_id)
    )
    existing_by_key = {
        (row.type, row.description): row for row in existing.scalars().all()
    }

    for inv in invariants:
        key = (InvariantType(inv.type), inv.description)
        evidence_uuids = _to_uuids(inv.evidence_trace_ids)

        if key in existing_by_key:
            row = existing_by_key[key]
            row.confidence = inv.confidence
            row.support_score = inv.support_score
            row.consistency_score = inv.consistency_score
            row.cross_val_bonus = inv.cross_val_bonus
            row.clarity_score = inv.clarity_score
            row.severity = Severity(inv.severity)
            row.status = InvariantStatus(inv.status)
            row.evidence_trace_ids = evidence_uuids
            row.evidence_count = inv.evidence_count
            row.discovered_by = inv.discovered_by
            row.params = inv.params
            continue

        session.add(
            Invariant(
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
                status=InvariantStatus(inv.status),
                evidence_trace_ids=evidence_uuids,
                evidence_count=inv.evidence_count,
                violation_count=0,
                discovered_by=inv.discovered_by,
                params=inv.params,
            )
        )


def _to_uuids(values: list[str]) -> list[uuid.UUID]:
    out: list[uuid.UUID] = []
    for v in values:
        try:
            out.append(uuid.UUID(str(v)))
        except (ValueError, TypeError):
            continue
    return out
