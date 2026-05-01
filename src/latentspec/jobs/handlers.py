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


# ---- cluster mining handler ---------------------------------------------


async def cluster_mining_job(ctx: JobContext, config: dict[str, Any]) -> dict[str, Any]:
    limit = int(config.get("limit", 2000))
    version_tag = config.get("version_tag")
    k_min = int(config.get("k_min", 2))
    k_max = int(config.get("k_max", 12))

    async with session_scope() as session:
        traces = await _load_traces(session, ctx.agent_id, limit=limit, version_tag=version_tag)
        await ctx.report_progress(10, f"loaded {len(traces)} traces")

        if not traces:
            return {"n_clusters": 0, "n_invariants": 0, "n_traces": 0}

        # Canonicalise upfront so cluster vectors don't see alias noise.
        cano = ToolCanonicalizer().fit(collect_tool_names(traces))
        await _persist_alias_decisions(session, ctx.agent_id, cano.decisions)
        traces = [apply_canonicalisation(t, cano) for t in traces]
        await ctx.report_progress(20, "canonicalisation done")

        result, clustering = await mine_per_cluster(
            agent_id=ctx.agent_id,
            traces=traces,
            k_min=k_min,
            k_max=k_max,
        )
        await ctx.report_progress(75, f"mined {result.n_clusters} clusters")

        # Persist centroids
        await session.execute(
            ClusterCentroid.__table__.delete().where(
                ClusterCentroid.agent_id == ctx.agent_id
            )
        )
        for cluster_id, n in result.cluster_sizes.items():
            session.add(
                ClusterCentroid(
                    agent_id=ctx.agent_id,
                    cluster_id=int(cluster_id),
                    centroid=clustering.centroids[int(cluster_id)].tolist(),
                    n_traces=int(n),
                    silhouette=float(result.silhouette),
                    vectorizer_state={
                        "tool_idx": dict(clustering.vectorizer._tool_idx),
                        "kw_idx": dict(clustering.vectorizer._kw_idx),
                    },
                )
            )
        await ctx.report_progress(95, "centroids persisted")

    return {
        "n_traces": result.n_traces,
        "n_clusters": result.n_clusters,
        "silhouette": result.silhouette,
        "cluster_sizes": result.cluster_sizes,
        "n_invariants": len(result.invariants),
    }


# ---- calibration handler ------------------------------------------------


async def calibration_job(ctx: JobContext, config: dict[str, Any]) -> dict[str, Any]:
    limit = int(config.get("limit", 2000))
    version_tag = config.get("version_tag")

    async with session_scope() as session:
        traces = await _load_traces(session, ctx.agent_id, limit=limit, version_tag=version_tag)
        await ctx.report_progress(15, f"loaded {len(traces)} traces")

        thresholds = calibrate_agent(traces)
        await ctx.report_progress(75, "calibration computed")

        existing = (
            await session.execute(
                select(CalibrationResult).where(
                    CalibrationResult.agent_id == ctx.agent_id
                )
            )
        ).scalar_one_or_none()
        if existing is None:
            existing = CalibrationResult(agent_id=ctx.agent_id)
            session.add(existing)

        existing.n_traces_calibrated = thresholds.distribution_summary.n_traces
        existing.mining_min_support = thresholds.mining_min_support
        existing.mining_min_directionality = thresholds.mining_min_directionality
        existing.mining_max_path_length = thresholds.mining_max_path_length
        existing.association_min_mi_bits = thresholds.association_min_mi_bits
        existing.association_min_lift = thresholds.association_min_lift
        existing.association_min_keyword_traces = thresholds.association_min_keyword_traces
        existing.statistical_p_target = thresholds.statistical_p_target
        existing.anomaly_contamination = thresholds.anomaly_contamination
        existing.confidence_reject_threshold = thresholds.confidence_reject_threshold
        existing.confidence_review_threshold = thresholds.confidence_review_threshold
        existing.fingerprint_chi_square_threshold = thresholds.fingerprint_chi_square_threshold
        existing.drift_ph_threshold = thresholds.drift_ph_threshold
        existing.drift_cusum_threshold = thresholds.drift_cusum_threshold
        existing.distribution_summary = {
            "n_traces": thresholds.distribution_summary.n_traces,
            "n_distinct_tools": thresholds.distribution_summary.n_distinct_tools,
            "n_distinct_fingerprints": thresholds.distribution_summary.n_distinct_fingerprints,
            "median_steps_per_trace": thresholds.distribution_summary.median_steps_per_trace,
            "p99_steps_per_trace": thresholds.distribution_summary.p99_steps_per_trace,
            "median_tool_calls_per_trace": thresholds.distribution_summary.median_tool_calls_per_trace,
            "pattern_support_curve": thresholds.distribution_summary.pattern_support_curve,
        }
        existing.calibrated_at = datetime.now(UTC)

    return {
        "n_traces_calibrated": thresholds.distribution_summary.n_traces,
        "thresholds": {
            "mining_min_support": thresholds.mining_min_support,
            "mining_max_path_length": thresholds.mining_max_path_length,
            "association_min_keyword_traces": thresholds.association_min_keyword_traces,
            "anomaly_contamination": thresholds.anomaly_contamination,
            "confidence_review_threshold": thresholds.confidence_review_threshold,
            "fingerprint_chi_square_threshold": thresholds.fingerprint_chi_square_threshold,
            "drift_ph_threshold": thresholds.drift_ph_threshold,
            "drift_cusum_threshold": thresholds.drift_cusum_threshold,
        },
    }


# ---- registration -------------------------------------------------------


def register_default_handlers() -> None:
    register_handler(JobKind.MINING, mining_job)
    register_handler(JobKind.CLUSTER_MINING, cluster_mining_job)
    register_handler(JobKind.CALIBRATION, calibration_job)
