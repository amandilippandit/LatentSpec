"""Default job handlers — wire all the orchestrators into the runner.

Each handler:
  - reads its agent's calibrated thresholds (when present)
  - applies tool canonicalisation to ingested traces
  - runs the orchestrator
  - persists results
  - reports progress to the JobContext
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from latentspec.calibration.calibrator import calibrate_agent
from latentspec.canonicalization.canonicalizer import (
    ToolCanonicalizer,
    collect_tool_names,
)
from latentspec.canonicalization.applier import apply_canonicalisation
from latentspec.db import session_scope
from latentspec.jobs.runner import JobContext, register_handler
from latentspec.mining.cluster_orchestrator import mine_per_cluster
from latentspec.mining.orchestrator import mine_invariants
from latentspec.models import (
    AgentVersion,
    CalibrationResult,
    ClusterCentroid,
    JobKind,
    Trace,
    ToolAlias,
)
from latentspec.schemas.trace import NormalizedTrace

log = logging.getLogger(__name__)


# ---- helpers ------------------------------------------------------------


async def _load_traces(
    session: AsyncSession, agent_id: uuid.UUID, *, limit: int, version_tag: str | None
) -> list[NormalizedTrace]:
    stmt = (
        select(Trace)
        .where(Trace.agent_id == agent_id)
        .order_by(Trace.started_at.desc())
        .limit(limit)
    )
    if version_tag is not None:
        stmt = stmt.where(Trace.version_tag == version_tag)
    rows = (await session.execute(stmt)).scalars().all()
    return [NormalizedTrace.model_validate(row.trace_data) for row in rows]


async def _alias_map(session: AsyncSession, agent_id: uuid.UUID) -> dict[str, str]:
    rows = (
        await session.execute(
            select(ToolAlias).where(ToolAlias.agent_id == agent_id)
        )
    ).scalars().all()
    return {row.raw_name: row.canonical_name for row in rows}


async def _persist_alias_decisions(
    session: AsyncSession, agent_id: uuid.UUID, decisions
) -> None:
    """Replace prior alias rows for this agent with the new decisions."""
    await session.execute(
        ToolAlias.__table__.delete().where(ToolAlias.agent_id == agent_id)
    )
    for d in decisions:
        session.add(
            ToolAlias(
                agent_id=agent_id,
                raw_name=d.raw_name,
                canonical_name=d.canonical_name,
                method=d.method,
                confidence=float(d.confidence),
            )
        )


# ---- mining handler -----------------------------------------------------


async def mining_job(ctx: JobContext, config: dict[str, Any]) -> dict[str, Any]:
    limit = int(config.get("limit", 1000))
    version_tag = config.get("version_tag")
    do_canonicalise = bool(config.get("canonicalise", True))
    do_persist = bool(config.get("persist", True))

    async with session_scope() as session:
        traces = await _load_traces(session, ctx.agent_id, limit=limit, version_tag=version_tag)
        await ctx.report_progress(10, f"loaded {len(traces)} traces")

        if do_canonicalise and traces:
            tool_names = collect_tool_names(traces)
            cano = ToolCanonicalizer().fit(tool_names)
            await _persist_alias_decisions(session, ctx.agent_id, cano.decisions)
            traces = [apply_canonicalisation(t, cano) for t in traces]
            await ctx.report_progress(
                25, f"canonicalised {len(tool_names)} → {len(cano.clusters)} tools"
            )

        result = await mine_invariants(
            agent_id=ctx.agent_id,
            traces=traces,
            session=session,
            persist=do_persist,
        )
        await ctx.report_progress(95, "mining complete")

    return {
        "traces_analyzed": result.traces_analyzed,
        "candidates_statistical": result.candidates_statistical,
        "candidates_llm": result.candidates_llm,
        "candidates_total_unique": result.candidates_total_unique,
        "by_status": result.by_status,
        "by_type": result.by_type,
        "duration_seconds": result.duration_seconds,
        "n_invariants": len(result.invariants),
        "mining_run_id": str(result.mining_run_id) if result.mining_run_id else None,
    }


