"""Tests for the §2.2 alert dispatcher (rate limiting + retry + signing)."""

from __future__ import annotations

import asyncio

import pytest

from latentspec.alerts.dispatcher import (
    AlertDispatcher,
    AlertEvent,
    AlertSink,
)


class FlakySink(AlertSink):
    """Fails N times then succeeds; records every attempt."""

    name = "flaky"

    def __init__(self, fail_first: int) -> None:
        self.fail_first = fail_first
        self.attempts: list[AlertEvent] = []
        self.successes = 0

    async def send(self, event: AlertEvent) -> None:
        self.attempts.append(event)
        if len(self.attempts) <= self.fail_first:
            raise RuntimeError("transient")
        self.successes += 1


@pytest.mark.asyncio
async def test_dispatcher_retries_then_succeeds() -> None:
    sink = FlakySink(fail_first=2)
    sink.base_backoff_s = 0.01  # speed up tests
    sink.max_attempts = 5

    d = AlertDispatcher()
    d.register(sink)

    await d.dispatch(_evt())
    assert sink.successes == 1
    assert len(sink.attempts) == 3  # 2 failures + 1 success


@pytest.mark.asyncio
async def test_dispatcher_gives_up_after_max_attempts() -> None:
    sink = FlakySink(fail_first=5)
    sink.base_backoff_s = 0.005
    sink.max_attempts = 3

    d = AlertDispatcher()
    d.register(sink)

    await d.dispatch(_evt())
    assert sink.successes == 0
    assert len(sink.attempts) == 3


@pytest.mark.asyncio
async def test_dispatcher_rate_limits_per_agent_per_sink() -> None:
    sink = FlakySink(fail_first=0)
    sink.rate_limit_per_min = 3

    d = AlertDispatcher()
    d.register(sink)

    for i in range(5):
        await d.dispatch(_evt(invariant_id=f"inv-{i}"))
    assert sink.successes == 3


@pytest.mark.asyncio
async def test_event_from_check_result_round_trip() -> None:
    from datetime import UTC, datetime

    from latentspec.checking.base import (
        CheckOutcome,
        CheckResult,
        ViolationDetails,
    )
    from latentspec.models.invariant import InvariantType, Severity

    cr = CheckResult(
        invariant_id="inv-1",
        invariant_type=InvariantType.ORDERING,
        invariant_description="rule",
        severity=Severity.HIGH,
        trace_id="t-1",
        outcome=CheckOutcome.FAIL,
        details=ViolationDetails(expected="A->B", observed="B->A"),
    )
    evt = AlertEvent.from_check_result(cr, agent_id="agent-1", agent_name="booking")
    assert evt.severity == "high"
    assert evt.observed == "B->A"


def _evt(*, invariant_id: str = "inv-1") -> AlertEvent:
    return AlertEvent(
        agent_id="agent-1",
        agent_name="booking",
        trace_id="t-1",
        invariant_id=invariant_id,
        invariant_description="rule",
        severity="critical",
        outcome="fail",
        observed="x",
    )
