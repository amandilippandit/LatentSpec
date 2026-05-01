"""Tests for the §5.1 SDK: client, decorators, and the trace() context."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

from latentspec.schemas.trace import (
    AgentResponseStep,
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)
from latentspec.sdk import client as client_mod
from latentspec.sdk.client import LatentSpecClient, SDKConfig
from latentspec.sdk.decorators import current_collector, trace, trace_tool


class FakeClient(LatentSpecClient):
    """Records traces in memory instead of shipping over the network."""

    def __init__(self) -> None:
        # Avoid the parent's network/thread setup — we just need .record()
        self._config = SDKConfig(api_key="ls_test", agent_id="agent-test", enabled=True)
        self._lock = threading.Lock()
        self.recorded: list[NormalizedTrace] = []

    @property
    def enabled(self) -> bool:  # type: ignore[override]
        return True

    def record(self, trace: NormalizedTrace) -> None:  # type: ignore[override]
        with self._lock:
            self.recorded.append(trace)

    def shutdown(self, timeout: float = 5.0) -> None:  # type: ignore[override]
        return

    def flush(self, timeout: float = 5.0) -> None:  # type: ignore[override]
        return


def setup_function() -> None:
    client_mod.configure_for_test(FakeClient())


def teardown_function() -> None:
    client_mod.configure_for_test(None)


def test_trace_context_emits_one_assembled_trace() -> None:
    fc = client_mod.get_client()
    assert isinstance(fc, FakeClient)

    with trace(user_input="book a flight"):
        # decorator captures a tool call into the active collector
        @trace_tool
        def search_flights(dest: str) -> dict:
            return {"flights": [dest]}

        search_flights("JFK")

    assert len(fc.recorded) == 1
    nt = fc.recorded[0]
    types = [s.type.value for s in nt.steps]
    assert "user_input" in types
    assert "tool_call" in types
    tool_step = next(s for s in nt.steps if isinstance(s, ToolCallStep))
    assert tool_step.tool == "search_flights"
    assert tool_step.args == {"dest": "JFK"}
    assert tool_step.result_status == "success"
    assert tool_step.latency_ms is not None and tool_step.latency_ms >= 0


def test_trace_tool_outside_context_emits_immediately() -> None:
    fc = client_mod.get_client()
    assert isinstance(fc, FakeClient)

    @trace_tool(name="weather")
    def get_weather(city: str) -> str:
        return f"{city}: sunny"

    get_weather("Tokyo")
    assert len(fc.recorded) == 1
    only_step = fc.recorded[0].steps[0]
    assert isinstance(only_step, ToolCallStep)
    assert only_step.tool == "weather"


def test_trace_context_records_response() -> None:
    fc = client_mod.get_client()
    assert isinstance(fc, FakeClient)

    with trace(user_input="hi") as t:
        t.add_response("hello back")

    nt = fc.recorded[0]
    assert any(isinstance(s, AgentResponseStep) for s in nt.steps)


def test_trace_tool_marks_error_status_on_exception() -> None:
    fc = client_mod.get_client()
    assert isinstance(fc, FakeClient)

    @trace_tool
    def boom() -> None:
        raise ValueError("nope")

    try:
        boom()
    except ValueError:
        pass

    only_step = fc.recorded[0].steps[0]
    assert isinstance(only_step, ToolCallStep)
    assert only_step.result_status == "error"


def test_current_collector_isolation() -> None:
    fc = client_mod.get_client()
    assert isinstance(fc, FakeClient)

    @trace_tool
    def step_a() -> None:
        return None

    assert current_collector() is None
    with trace(user_input="x"):
        assert current_collector() is not None
        step_a()
    assert current_collector() is None
