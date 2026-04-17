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
        # Local-cache fast path
        invariants = self._cache.get_local(agent_id)
        if invariants is None:
            invariants = await self._cache.get_or_load(agent_id, loader=loader)

        checked = 0
        passed = 0
        warned = 0
        failed = 0
        failures: list[CheckResult] = []
        warnings: list[CheckResult] = []
        fail_open = False
        drift_collected: list[DriftEvent] = []

        deadline = start + (self._budget_ms / 1000.0)

        for inv in invariants:
            now = time.perf_counter()
            if now >= deadline:
                # Fail-open: surface what we checked, mark fail_open=True.
                fail_open = True
                break

            try:
                result = dispatch(inv, trace)
            except Exception as e:  # noqa: BLE001
                log.debug("checker error on %s: %s", inv.id, e)
                continue

            checked += 1
            if result.outcome == CheckOutcome.PASS:
                passed += 1
            elif result.outcome == CheckOutcome.NOT_APPLICABLE:
                pass
            elif result.outcome == CheckOutcome.FAIL:
                failed += 1
                failures.append(result)
            elif result.outcome == CheckOutcome.WARN:
                warned += 1
                warnings.append(result)

            # Online drift detection — feed pass/fail outcomes to PH+CUSUM
            if result.outcome in (CheckOutcome.PASS, CheckOutcome.FAIL):
                drift_events_raw = self._drift.observe(
                    agent_id, inv.id, result.outcome == CheckOutcome.PASS
                )
                if drift_events_raw:
                    # Lazily allocate the result list field (default is [])
                    drift_collected.extend(drift_events_raw)

        duration_ms = round((time.perf_counter() - start) * 1000, 3)
        with self._lock:
            self._latencies.append(duration_ms)
            if fail_open:
                self._fail_open_count += 1

        sr = StreamingResult(
            trace_id=trace.trace_id,
            agent_id=agent_id,
            invariants_checked=checked,
            passed=passed,
            warned=warned,
            failed=failed,
            duration_ms=duration_ms,
            failures=failures,
            warnings=warnings,
            fail_open=fail_open,
            drift_events=drift_collected,
        )

        if (failures or warnings or drift_collected) and self._on_violation is not None:
            try:
                await self._on_violation(sr)
            except Exception as e:  # noqa: BLE001
                log.warning("violation hook failed: %s", e)

        return sr

    def stats(self) -> DetectionStats | None:
        """Return rolling latency / fail-open metrics for ops dashboards."""
        with self._lock:
            data = list(self._latencies)
            failo = self._fail_open_count
        if not data:
            return None
        sorted_data = sorted(data)
        n = len(sorted_data)
        return DetectionStats(
            p50_ms=sorted_data[int(n * 0.5)],
            p95_ms=sorted_data[min(n - 1, int(n * 0.95))],
            p99_ms=sorted_data[min(n - 1, int(n * 0.99))],
            sample_size=n,
            fail_open_rate=round(failo / max(1, n), 4),
        )
