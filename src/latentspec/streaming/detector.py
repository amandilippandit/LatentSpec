"""Streaming detector — sub-100ms inline checks with circuit breaking.

For each incoming trace we:
  1. Pull the active invariant set from the hot cache (sync, <1ms).
  2. Run every applicable rule-based checker (synchronous, <100ms p99).
  3. Classify the trace into pass / warn / fail buckets.
  4. Emit alerts for FAIL outcomes via the alert dispatcher.

The detector is *fail-open*: if checking takes too long, we surface a
WARN-level health metric rather than blocking the request path. Critical
gating is the GitHub Action's job (§4.2), not the streaming hot path.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from latentspec.checking.base import (
    CheckOutcome,
    CheckResult,
    InvariantSpec,
)
from latentspec.checking.dispatch import dispatch
from latentspec.schemas.trace import NormalizedTrace
from latentspec.streaming.cache import HotInvariantCache, get_cache
from latentspec.streaming.drift import DriftEvent, DriftRegistry, get_drift_registry

log = logging.getLogger(__name__)


@dataclass
class StreamingResult:
    trace_id: str
    agent_id: str
    invariants_checked: int
    passed: int
    warned: int
    failed: int
    duration_ms: float
    failures: list[CheckResult] = field(default_factory=list)
    warnings: list[CheckResult] = field(default_factory=list)
    fail_open: bool = False
    drift_events: list[DriftEvent] = field(default_factory=list)


@dataclass
class DetectionStats:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    sample_size: int
    fail_open_rate: float


class StreamingDetector:
    """Single-trace fast-path detector with rolling latency stats.

    Spawn one per process. The detector itself is stateless beyond a fixed-
    size latency ring buffer used for runtime health metrics.
    """

    def __init__(
        self,
        *,
        cache: HotInvariantCache | None = None,
        check_budget_ms: int = 100,
        latency_window: int = 1024,
        on_violation: Callable[[StreamingResult], Awaitable[None]] | None = None,
        drift_registry: DriftRegistry | None = None,
    ) -> None:
        self._cache = cache or get_cache()
        self._budget_ms = check_budget_ms
        self._latencies: deque[float] = deque(maxlen=latency_window)
        self._fail_open_count = 0
        self._lock = threading.Lock()
        self._on_violation = on_violation
        self._drift = drift_registry or get_drift_registry()

    async def check(
        self,
        agent_id: str,
        trace: NormalizedTrace,
        *,
        loader: Callable[[str], Awaitable[list[InvariantSpec]]],
    ) -> StreamingResult:
        """Check one trace against the agent's active invariant set."""
        start = time.perf_counter()
