"""Isolation-forest anomaly baselines (§3.2 Track A).

Builds a per-agent behavioral feature vector for every trace and runs
sklearn's IsolationForest to surface outliers. Each surfaced cluster of
similar anomalies becomes a candidate STATISTICAL invariant of the form
"behavioral feature X stays within the observed normal range".

The features are deliberately structural — counts, latencies, distinct-
tool counts — so the invariants the miner produces are directly checkable
by the runtime detector without needing the trained model at check time.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.ensemble import IsolationForest

from latentspec.models.invariant import InvariantType, Severity
from latentspec.schemas.invariant import InvariantCandidate
from latentspec.schemas.trace import (
    AgentResponseStep,
    NormalizedTrace,
    StepType,
    ToolCallStep,
    UserInputStep,
)


_FEATURE_NAMES = [
    "step_count",
    "tool_call_count",
    "distinct_tool_count",
    "user_input_chars",
    "agent_response_chars",
    "total_latency_ms",
    "max_step_latency_ms",
    "error_step_count",
]


def _features(trace: NormalizedTrace) -> list[float]:
    tool_calls = [s for s in trace.steps if isinstance(s, ToolCallStep)]
    distinct = {s.tool for s in tool_calls}
    user_input_chars = sum(
        len(s.content) for s in trace.steps if isinstance(s, UserInputStep)
    )
    agent_response_chars = sum(
        len(s.content) for s in trace.steps if isinstance(s, AgentResponseStep)
    )
    latencies = [s.latency_ms or 0 for s in tool_calls]
    error_count = sum(
        1 for s in tool_calls if (s.result_status or "success") != "success"
    )
    return [
        float(len(trace.steps)),
        float(len(tool_calls)),
        float(len(distinct)),
        float(user_input_chars),
        float(agent_response_chars),
        float(sum(latencies)),
        float(max(latencies) if latencies else 0),
        float(error_count),
    ]


def mine_anomaly_baselines(
    traces: list[NormalizedTrace],
    *,
    min_traces: int = 30,
