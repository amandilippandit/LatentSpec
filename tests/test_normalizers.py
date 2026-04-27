"""Tests for Stage 1 normalizers (raw JSON + LangChain)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from latentspec.normalizers import LangChainNormalizer, NormalizerError, RawJSONNormalizer
from latentspec.schemas.trace import (
    AgentResponseStep,
    NormalizedTrace,
    StepType,
    ToolCallStep,
    UserInputStep,
)


def test_raw_json_accepts_minimal_payload() -> None:
    payload = {
        "steps": [
            {"type": "user_input", "content": "hi"},
            {"type": "agent_response", "content": "hello"},
        ]
    }
    nt = RawJSONNormalizer().normalize(payload, agent_id="agent-1")
    assert isinstance(nt, NormalizedTrace)
    assert nt.agent_id == "agent-1"
    assert len(nt.steps) == 2
    assert nt.steps[0].type == StepType.USER_INPUT
    assert nt.steps[1].type == StepType.AGENT_RESPONSE


def test_raw_json_rejects_missing_steps() -> None:
    with pytest.raises(NormalizerError):
        RawJSONNormalizer().normalize({"trace_id": "x"}, agent_id="agent-1")


def test_raw_json_round_trip() -> None:
    original = NormalizedTrace(
        trace_id="trace-1",
        agent_id="a",
        timestamp=datetime.now(UTC),
        steps=[
            UserInputStep(content="book a flight"),
            ToolCallStep(tool="search_flights", args={"q": "JFK"}, latency_ms=210),
            AgentResponseStep(content="found 3 flights"),
        ],
    )
    payload = original.model_dump(mode="json")
    nt = RawJSONNormalizer().normalize(payload, agent_id="a")
    assert nt.trace_id == "trace-1"
    assert len(nt.steps) == 3


def test_langchain_extracts_user_input_and_response() -> None:
    payload = {
        "id": "lc-1",
        "start_time": "2026-01-01T00:00:00Z",
        "end_time": "2026-01-01T00:00:02Z",
        "run_type": "chain",
        "inputs": {"input": "search flights to Tokyo"},
        "outputs": {"output": "found 3 options"},
        "child_runs": [
            {
                "run_type": "tool",
                "name": "search_flights",
                "start_time": "2026-01-01T00:00:00.500Z",
                "end_time": "2026-01-01T00:00:01.0Z",
                "inputs": {"dest": "NRT"},
                "outputs": {"flights": ["NH101", "JL5", "DL7"]},
            }
        ],
    }
    nt = LangChainNormalizer().normalize(payload, agent_id="agent-x")
    assert nt.trace_id == "lc-1"
    types = [s.type for s in nt.steps]
    assert StepType.USER_INPUT in types
    assert StepType.TOOL_CALL in types
    assert StepType.AGENT_RESPONSE in types
    tool_step = next(s for s in nt.steps if isinstance(s, ToolCallStep))
    assert tool_step.tool == "search_flights"
    assert tool_step.latency_ms is not None
    assert tool_step.latency_ms >= 0


def test_langchain_rejects_empty_payload() -> None:
    with pytest.raises(NormalizerError):
        LangChainNormalizer().normalize(
            {"id": "x", "run_type": "chain", "inputs": {}, "outputs": {}},
            agent_id="agent-x",
        )
