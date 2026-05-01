"""Tests for the §4.1 streaming detector + hot invariant cache."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from latentspec.checking.base import InvariantSpec
from latentspec.models.invariant import InvariantType, Severity
from latentspec.schemas.trace import (
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)
from latentspec.streaming.cache import InMemoryCache
from latentspec.streaming.detector import StreamingDetector


def _trace(steps) -> NormalizedTrace:
    return NormalizedTrace(
        trace_id="t",
        agent_id="agent-1",
        timestamp=datetime.now(UTC),
        steps=list(steps),
        metadata=TraceMetadata(),
    )


def _ordering_rule() -> InvariantSpec:
    return InvariantSpec(
        id="inv-ord",
        type=InvariantType.ORDERING,
        description="auth before send_email",
        formal_rule="placeholder",
        severity=Severity.CRITICAL,
        params={"tool_a": "auth", "tool_b": "send_email"},
    )


@pytest.mark.asyncio
async def test_detector_passes_clean_trace() -> None:
    cache = InMemoryCache()
    await cache.warm("agent-1", [_ordering_rule()])
    det = StreamingDetector(cache=cache)

    async def loader(_: str):
        return [_ordering_rule()]

    sr = await det.check(
        "agent-1",
        _trace(
            [
                ToolCallStep(tool="auth", args={}),
                ToolCallStep(tool="send_email", args={}),
            ]
        ),
        loader=loader,
    )
    assert sr.failed == 0
    assert sr.passed == 1
    assert sr.duration_ms < 100, f"latency budget breached: {sr.duration_ms}ms"


@pytest.mark.asyncio
async def test_detector_flags_violation_and_hits_violation_hook() -> None:
    cache = InMemoryCache()
    await cache.warm("agent-1", [_ordering_rule()])
    seen: list = []

    async def hook(sr) -> None:  # noqa: ANN001
        seen.append(sr)

    det = StreamingDetector(cache=cache, on_violation=hook)

    async def loader(_: str):
        return [_ordering_rule()]

    sr = await det.check(
        "agent-1",
        _trace(
            [
                ToolCallStep(tool="send_email", args={}),
                ToolCallStep(tool="auth", args={}),
            ]
        ),
        loader=loader,
    )
    assert sr.failed == 1
    assert seen, "violation hook should have been called"


@pytest.mark.asyncio
async def test_cache_warm_makes_local_cache_hit_immediate() -> None:
    cache = InMemoryCache(ttl_seconds=60.0)
    await cache.warm("agent-1", [_ordering_rule()])
    assert cache.get_local("agent-1") is not None


@pytest.mark.asyncio
async def test_detector_records_latency_stats() -> None:
    cache = InMemoryCache()
    await cache.warm("agent-1", [_ordering_rule()])
    det = StreamingDetector(cache=cache)

    async def loader(_: str):
        return [_ordering_rule()]

    for _ in range(10):
        await det.check(
            "agent-1",
            _trace(
                [
                    UserInputStep(content="hi"),
                    ToolCallStep(tool="auth", args={}),
                    ToolCallStep(tool="send_email", args={}),
                ]
            ),
            loader=loader,
        )
    stats = det.stats()
    assert stats is not None
    assert stats.sample_size == 10
    assert stats.p99_ms < 100, f"p99 budget breached: {stats.p99_ms}ms"
