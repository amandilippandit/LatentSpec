"""Read-only observability endpoints — Prometheus metrics + detector stats."""

from __future__ import annotations

from fastapi import APIRouter, Response

from latentspec.api.routes.traces import _detector
from latentspec.observability.metrics import metrics_text

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(content=metrics_text(), media_type="text/plain; version=0.0.4")


@router.get("/streaming/stats")
async def streaming_stats() -> dict:
    """Return rolling p50/p95/p99 latency + fail-open rate for the detector."""
    s = _detector.stats()
    if s is None:
        return {"sample_size": 0}
    return {
        "p50_ms": s.p50_ms,
        "p95_ms": s.p95_ms,
        "p99_ms": s.p99_ms,
        "sample_size": s.sample_size,
        "fail_open_rate": s.fail_open_rate,
    }
