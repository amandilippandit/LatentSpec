"""Tests for Page-Hinkley + CUSUM drift detection in the streaming layer."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from latentspec.checking.base import InvariantSpec
from latentspec.models.invariant import InvariantType, Severity
from latentspec.schemas.trace import NormalizedTrace, ToolCallStep, TraceMetadata
from latentspec.streaming.cache import InMemoryCache
from latentspec.streaming.detector import StreamingDetector
from latentspec.streaming.drift import (
    CusumDetector,
    DriftRegistry,
    PageHinkleyDetector,
)


def test_page_hinkley_fires_on_pass_rate_drop() -> None:
    ph = PageHinkleyDetector(threshold=2.0, delta=0.001, min_samples=10)
    # 100 passes — running mean stabilises near 1.0
    for _ in range(100):
        assert not ph.update(1.0)
    # Then a stretch of failures — pass rate drops, cumsum diverges
    fired = False
    for _ in range(60):
        if ph.update(0.0):
            fired = True
            break
    assert fired, "PH should have fired on the failure run"


def test_page_hinkley_does_not_fire_on_steady_state() -> None:
    ph = PageHinkleyDetector(threshold=8.0, delta=0.005, min_samples=30)
    # Mostly passing with rare random failures (5%) shouldn't trip drift.
    import random

    rng = random.Random(7)
    fired = False
    for _ in range(500):
        v = 0.0 if rng.random() < 0.05 else 1.0
        if ph.update(v):
            fired = True
            break
    assert not fired


def test_cusum_detects_upward_shift() -> None:
    cs = CusumDetector(target=0.5, slack=0.05, threshold=2.0, min_samples=10)
    # First 50 around target 0.5
    import random

    rng = random.Random(11)
    for _ in range(50):
        cs.update(0.5 + (rng.random() - 0.5) * 0.05)
    fired = False
    # Then jump to 0.95
    for _ in range(40):
        if cs.update(0.95):
            fired = True
            break
    assert fired
    assert cs.direction == "up"


def test_drift_registry_isolates_per_invariant() -> None:
