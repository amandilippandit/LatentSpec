"""Tests for behavioral fingerprinting + drift on the fingerprint distribution."""

from __future__ import annotations

from datetime import UTC, datetime

from latentspec.mining.fingerprint import (
    FingerprintDistribution,
    canonical_shape,
    fingerprint,
    fingerprint_set,
)
from latentspec.schemas.trace import (
    AgentResponseStep,
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)


def _trace(steps) -> NormalizedTrace:
    return NormalizedTrace(
        trace_id="t",
        agent_id="a",
        timestamp=datetime.now(UTC),
        steps=list(steps),
        metadata=TraceMetadata(),
    )


def test_canonical_shape_ignores_args_and_content() -> None:
    a = _trace(
        [
            UserInputStep(content="book a flight to Tokyo"),
            ToolCallStep(tool="search_flights", args={"dest": "NRT"}),
            AgentResponseStep(content="found 3 flights"),
        ]
    )
    b = _trace(
        [
            UserInputStep(content="book a flight to Paris"),
            ToolCallStep(tool="search_flights", args={"dest": "CDG"}),
            AgentResponseStep(content="found 5 flights"),
        ]
    )
    assert canonical_shape(a) == canonical_shape(b)
    assert fingerprint(a) == fingerprint(b)


def test_different_tool_sequences_have_different_fingerprints() -> None:
    a = _trace(
        [
            UserInputStep(content="hi"),
            ToolCallStep(tool="auth", args={}),
            ToolCallStep(tool="db_write", args={}),
        ]
    )
    b = _trace(
        [
            UserInputStep(content="hi"),
            ToolCallStep(tool="db_write", args={}),
            ToolCallStep(tool="auth", args={}),
        ]
    )
    assert fingerprint(a) != fingerprint(b)


def test_distribution_detects_chi_square_drift() -> None:
    a = _trace([ToolCallStep(tool="auth", args={})])
    b = _trace([ToolCallStep(tool="search", args={})])

    dist = FingerprintDistribution()
    for _ in range(80):
        dist.add(fingerprint(a))
    for _ in range(20):
        dist.add(fingerprint(b))
    dist.update_baseline()
    dist.reset_observation_window()

    # Now post-baseline traffic flips the mix
    for _ in range(20):
        dist.add(fingerprint(a))
    for _ in range(80):
        dist.add(fingerprint(b))

    assert dist.is_drifting()


def test_distribution_quiet_when_mix_stable() -> None:
    a = _trace([ToolCallStep(tool="auth", args={})])
    b = _trace([ToolCallStep(tool="search", args={})])
    import random

    rng = random.Random(7)
    dist = FingerprintDistribution()
    for _ in range(200):
        dist.add(fingerprint(a if rng.random() < 0.6 else b))
    dist.update_baseline()
    dist.reset_observation_window()
    for _ in range(200):
        dist.add(fingerprint(a if rng.random() < 0.6 else b))
    assert not dist.is_drifting()


def test_fingerprint_set_returns_multiset() -> None:
    a = _trace([ToolCallStep(tool="auth", args={})])
    b = _trace([ToolCallStep(tool="search", args={})])
    counts = fingerprint_set([a, a, b])
    assert sum(counts.values()) == 3
    assert len(counts) == 2
