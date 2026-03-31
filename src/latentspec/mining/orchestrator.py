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
