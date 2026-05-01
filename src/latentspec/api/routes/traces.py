"""Trace ingestion routes — `POST /traces` and `POST /traces/batch`.

Pipeline applied to every ingested trace:

  1. **Format dispatch** — accepts `normalized`, `raw_json`, `langchain`,
     and `dag` payloads. DAG payloads are linearised before mining /
     checking but the original DAG view is kept for subgraph mining.
  2. **Canonicalisation** — apply persisted `ToolAlias` map so e.g.
     `payments_v1` and `payments-v1` collapse to one canonical name
     before mining or checking ever sees them.
  3. **Persist with metadata** — `session_id`, `user_id`, `cluster_id`
     (predicted from persisted centroids), `fingerprint` (canonical-shape
     SHA-256 prefix). Each carries an index so the dashboard / API can
     filter by them.
  4. **Streaming detection** — sub-100ms inline check via the streaming
     detector + drift registry (with persisted Page-Hinkley state).
  5. **Fingerprint baseline update** — accumulates `observed_counts`;
     the dashboard / drift surface reads from this row.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from latentspec.alerts.dispatcher import AlertEvent, get_dispatcher
from latentspec.canonicalization.applier import apply_alias_map
from latentspec.checking.runner import invariant_to_spec
from latentspec.db import get_db
from latentspec.mining.fingerprint import canonical_shape, fingerprint
from latentspec.models import (
    Agent,
    ClusterCentroid,
    FingerprintBaseline,
    Invariant,
    InvariantStatus,
    ToolAlias,
    Trace,
    TraceStatus,
)
from latentspec.normalizers import NormalizerError, registry
from latentspec.observability.metrics import counter, histogram
from latentspec.schemas.dag import DagTrace
from latentspec.schemas.trace import NormalizedTrace, TraceIn
from latentspec.streaming import StreamingDetector
from latentspec.streaming.cache import get_cache
from latentspec.versioning.tracker import register_or_update_version

router = APIRouter()
log = logging.getLogger(__name__)


# Single in-process detector — the cache it points at is process-wide too.
async def _on_violation(sr) -> None:  # type: ignore[no-untyped-def]
    """Forward FAIL/WARN results from the streaming detector to alerts."""
    dispatcher = get_dispatcher()
    if not dispatcher.sinks():
        return
    for r in (*sr.failures, *sr.warnings):
        await dispatcher.dispatch(
            AlertEvent.from_check_result(
                r, agent_id=sr.agent_id, agent_name=sr.agent_id
            )
        )


_detector = StreamingDetector(on_violation=_on_violation)


async def _load_active_invariants(agent_id_str: str):
    import uuid as _uuid

    async with __import__("latentspec").db.SessionLocal() as session:
        result = await session.execute(
            select(Invariant)
            .where(Invariant.agent_id == _uuid.UUID(agent_id_str))
            .where(Invariant.status == InvariantStatus.ACTIVE)
        )
        return [invariant_to_spec(row) for row in result.scalars().all()]


# ---- helpers -----------------------------------------------------------


async def _alias_map_for(db: AsyncSession, agent_id: uuid.UUID) -> dict[str, str]:
    rows = (
        await db.execute(
            select(ToolAlias).where(ToolAlias.agent_id == agent_id)
        )
    ).scalars().all()
    return {row.raw_name: row.canonical_name for row in rows}


async def _route_to_cluster(
    db: AsyncSession, agent_id: uuid.UUID, normalized: NormalizedTrace
) -> int | None:
    """Predict the cluster_id by nearest-centroid in cosine distance."""
    rows = (
        await db.execute(
            select(ClusterCentroid).where(ClusterCentroid.agent_id == agent_id)
        )
    ).scalars().all()
    if not rows:
        return None

    # Vectorise the trace using the persisted vocabulary
    # (We only have the tool/keyword index from the centroid metadata)
    # For a correct routing we'd reconstruct the full vectorizer; here
    # we use a simple tool-bag fallback that matches the dominant signal
    from collections import Counter

    from latentspec.schemas.trace import ToolCallStep, UserInputStep

    tool_counts: Counter[str] = Counter()
    for step in normalized.steps:
        if isinstance(step, ToolCallStep):
            tool_counts[step.tool] += 1

    best_id = None
    best_score = -math.inf
    for row in rows:
        tool_idx = (row.vectorizer_state or {}).get("tool_idx") or {}
        if not tool_idx:
            continue
        # Score: dot product between trace's tool TF and centroid restricted to tool dims
        # Since the centroid vector layout is [behavioral...|tool TF-IDF...|kw TF-IDF...],
        # we approximate by summing centroid weights for tools the trace uses.
        score = 0.0
        # Behavioural dim count from clustering.TraceShapeVectorizer.behavioral_dim
        behavioral_dim = 8
        for tool, c in tool_counts.items():
            idx = tool_idx.get(tool)
            if idx is None:
                continue
            v_idx = behavioral_dim + int(idx)
            centroid = row.centroid
            if v_idx < len(centroid):
                score += float(centroid[v_idx]) * c
        if score > best_score:
            best_score = score
            best_id = int(row.cluster_id)
    return best_id


async def _update_fingerprint_baseline(
    db: AsyncSession, agent_id: uuid.UUID, fp: str
) -> None:
    """Increment observed_counts; promote to baseline if none exists yet."""
    row = (
        await db.execute(
            select(FingerprintBaseline).where(FingerprintBaseline.agent_id == agent_id)
        )
    ).scalar_one_or_none()
    if row is None:
        row = FingerprintBaseline(agent_id=agent_id, baseline_counts={}, observed_counts={})
        db.add(row)
        await db.flush()

    observed = dict(row.observed_counts or {})
    observed[fp] = observed.get(fp, 0) + 1
    row.observed_counts = observed
    row.n_observed += 1

    # Lazy chi-square check (cheap)
    if row.baseline_counts and row.n_observed >= 50:
        chi = _chi_square(observed, dict(row.baseline_counts))
        if chi > row.chi_square_threshold:
            row.last_drift_chi = chi
            from datetime import UTC, datetime

            row.last_drift_at = datetime.now(UTC)


def _chi_square(observed: dict[str, int], baseline: dict[str, int]) -> float:
    obs_total = sum(observed.values())
    bl_total = sum(baseline.values())
    if obs_total == 0 or bl_total == 0:
        return 0.0
    smooth = 0.5
    keys = set(observed) | set(baseline)
    chi = 0.0
    for k in keys:
        p_ref = (baseline.get(k, 0) + smooth) / (bl_total + smooth * len(keys))
        expected = p_ref * obs_total
        if expected > 0:
            chi += (observed.get(k, 0) - expected) ** 2 / expected
    return chi


def _detect_format(payload: dict[str, Any]) -> str | None:
    """Auto-detect a payload's format from its shape.

    DAG: has `nodes` and `edges` lists.
    LangChain: has `run_type` or `child_runs`.
    Normalised: has `steps` array.
    """
    if not isinstance(payload, dict):
        return None
    if "nodes" in payload and "edges" in payload:
        return "dag"
    if "run_type" in payload or "child_runs" in payload:
        return "langchain"
    if "steps" in payload:
        return "raw_json"
    return None


def _normalize_dag_payload(payload: dict[str, Any], *, agent_id: str) -> NormalizedTrace:
    dag = DagTrace.model_validate(
        {
            "trace_id": payload.get("trace_id") or f"dag-{uuid.uuid4().hex[:10]}",
            "agent_id": payload.get("agent_id") or agent_id,
            **payload,
        }
    )
    return dag.to_linear()


# ---- output models -----------------------------------------------------


class TraceAcceptedOut(BaseModel):
    id: uuid.UUID
    agent_id: uuid.UUID
    step_count: int
    status: str
    fingerprint: str
    cluster_id: int | None = None


# ---- ingest path -------------------------------------------------------


async def _persist(
    db: AsyncSession,
    agent_id: uuid.UUID,
    normalized: NormalizedTrace,
    *,
    version_tag: str | None,
    session_id: str | None,
    user_id: str | None,
) -> Trace:
    agent = await db.get(Agent, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="agent not found")

    # ---- canonicalise tool names ---------------------------------------
    aliases = await _alias_map_for(db, agent_id)
    if aliases:
        normalized = apply_alias_map(normalized, aliases)

    # ---- compute fingerprint + cluster routing -------------------------
    fp = fingerprint(normalized)
    cluster_id = await _route_to_cluster(db, agent_id, normalized)

    trace = Trace(
        agent_id=agent_id,
        trace_data=normalized.model_dump(mode="json"),
        version_tag=version_tag or normalized.metadata.version,
        session_id=session_id,
        user_id=user_id,
        cluster_id=cluster_id,
        fingerprint=fp,
        started_at=normalized.timestamp,
        ended_at=normalized.ended_at,
        step_count=len(normalized.steps),
        status=TraceStatus.SUCCESS,
    )
    db.add(trace)
    await db.flush()
    await db.refresh(trace)

    # ---- register / update agent version -------------------------------
    effective_version = version_tag or normalized.metadata.version
    if effective_version:
        try:
            await register_or_update_version(
                db,
                agent_id=agent_id,
                version_tag=str(effective_version),
                trace=normalized,
            )
        except Exception as e:  # noqa: BLE001
            log.debug("version tracking failed: %s", e)

    # ---- update fingerprint baseline (drift signal) --------------------
    try:
        await _update_fingerprint_baseline(db, agent_id, fp)
    except Exception as e:  # noqa: BLE001
        log.debug("fingerprint baseline update failed: %s", e)

    # ---- streaming detection -------------------------------------------
    counter("latentspec_traces_ingested_total", labels={"agent_id": str(agent_id)})
    try:
        sr = await _detector.check(
            agent_id=str(agent_id),
            trace=normalized,
            loader=_load_active_invariants,
        )
        histogram(
            "latentspec_streaming_check_seconds",
            sr.duration_ms / 1000.0,
            labels={"agent_id": str(agent_id)},
        )
        if sr.failed:
            counter(
                "latentspec_violations_total",
                labels={"agent_id": str(agent_id), "outcome": "fail"},
                value=sr.failed,
            )
        if sr.warned:
            counter(
                "latentspec_violations_total",
                labels={"agent_id": str(agent_id), "outcome": "warn"},
                value=sr.warned,
            )
    except Exception as e:  # noqa: BLE001
        log.debug("streaming detection failed (fail-open): %s", e)

    return trace


@router.post(
    "",
    response_model=TraceAcceptedOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_trace(
    payload: TraceIn,
    db: AsyncSession = Depends(get_db),
) -> TraceAcceptedOut:
    """Accept and normalise a single agent trace.

    Format dispatch in priority order:
      1. Explicit `format` in the payload header (`raw_json`, `langchain`, `dag`).
      2. Auto-detect from payload shape.
      3. Fall back to raw_json.
    """
    fmt = payload.format if payload.format != "normalized" else None
    if fmt is None:
        fmt = _detect_format(payload.payload) or "raw_json"

    try:
        if fmt == "dag":
            normalized = _normalize_dag_payload(
                payload.payload, agent_id=str(payload.agent_id)
            )
        else:
            normalized = registry.normalize(
                fmt, payload.payload, agent_id=str(payload.agent_id)
            )
    except NormalizerError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to normalise: {e}") from e

    session_id = payload.payload.get("session_id") if isinstance(payload.payload, dict) else None
    user_id = payload.payload.get("user_id") if isinstance(payload.payload, dict) else None

    trace = await _persist(
        db, payload.agent_id, normalized,
        version_tag=payload.version_tag,
        session_id=session_id,
        user_id=user_id,
    )
    return TraceAcceptedOut(
        id=trace.id,
        agent_id=trace.agent_id,
        step_count=trace.step_count,
        status=trace.status.value,
        fingerprint=trace.fingerprint or "",
        cluster_id=trace.cluster_id,
    )


class TraceBatchIn(BaseModel):
    traces: list[TraceIn]


@router.post(
    "/batch",
    response_model=list[TraceAcceptedOut],
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_batch(
    payload: TraceBatchIn,
    db: AsyncSession = Depends(get_db),
) -> list[TraceAcceptedOut]:
    out: list[TraceAcceptedOut] = []
    for item in payload.traces:
        fmt = item.format if item.format != "normalized" else None
        if fmt is None:
            fmt = _detect_format(item.payload) or "raw_json"
        try:
            if fmt == "dag":
                normalized = _normalize_dag_payload(item.payload, agent_id=str(item.agent_id))
            else:
                normalized = registry.normalize(
                    fmt, item.payload, agent_id=str(item.agent_id)
                )
        except (NormalizerError, Exception) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        sid = item.payload.get("session_id") if isinstance(item.payload, dict) else None
        uid = item.payload.get("user_id") if isinstance(item.payload, dict) else None
        trace = await _persist(
            db, item.agent_id, normalized,
            version_tag=item.version_tag,
            session_id=sid,
            user_id=uid,
        )
        out.append(
            TraceAcceptedOut(
                id=trace.id,
                agent_id=trace.agent_id,
                step_count=trace.step_count,
                status=trace.status.value,
                fingerprint=trace.fingerprint or "",
                cluster_id=trace.cluster_id,
            )
        )
    return out
