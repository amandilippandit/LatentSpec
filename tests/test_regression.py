"""Tests for §4.1 batch comparison + §4.2 PR-comment renderer."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

import pytest

from latentspec.checking.base import InvariantSpec
from latentspec.demo import generate_traces
from latentspec.mining.orchestrator import mine_invariants
from latentspec.models.invariant import InvariantType, Severity
from latentspec.regression.batch import _exit_code_for, compare_trace_sets
from latentspec.regression.report import format_pr_comment
from latentspec.schemas.trace import (
    NormalizedTrace,
    ToolCallStep,
    UserInputStep,
)


def _trace(idx: int, steps) -> NormalizedTrace:
    return NormalizedTrace(
        trace_id=f"t-{idx}",
        agent_id="agent",
        timestamp=datetime.now(UTC),
        steps=list(steps),
    )


def _spec(type_, params, severity=Severity.CRITICAL) -> InvariantSpec:
    return InvariantSpec(
        id="inv-test",
        type=type_,
        description="Agent always calls `auth` before `db_write`",
        formal_rule="forall trace: ...",
        severity=severity,
        params=params,
    )


def test_baseline_passes_candidate_regresses() -> None:
    inv = _spec(InvariantType.ORDERING, {"tool_a": "auth", "tool_b": "db_write"})
    baseline = [
        _trace(
            i,
            [
                UserInputStep(content="hi"),
                ToolCallStep(tool="auth", args={}),
                ToolCallStep(tool="db_write", args={}),
            ],
        )
        for i in range(20)
    ]
    candidate = [
        _trace(
            i,
            [
                UserInputStep(content="hi"),
                ToolCallStep(tool="db_write", args={}),  # auth removed
            ],
        )
        for i in range(20)
    ]
    report = compare_trace_sets([inv], baseline, candidate)
    assert report.passes == 0
    assert len(report.failures) == 1
    assert report.failures[0].fail_rate == 1.0
    assert _exit_code_for(report, "critical") == 1
    body = format_pr_comment(report, agent_name="booking-agent")
    assert "FAIL" in body
    assert "CRITICAL" in body or "critical" in body.lower()


def test_passing_candidate_zero_failures() -> None:
    inv = _spec(InvariantType.ORDERING, {"tool_a": "auth", "tool_b": "db_write"})
    traces = [
        _trace(
            i,
            [
                ToolCallStep(tool="auth", args={}),
                ToolCallStep(tool="db_write", args={}),
            ],
        )
        for i in range(20)
    ]
    report = compare_trace_sets([inv], traces, traces)
    assert report.failures == []
    assert report.passes == 1
    assert _exit_code_for(report, "critical") == 0


@pytest.mark.asyncio
async def test_intentional_regression_demo_flow() -> None:
    """Mirrors the §9 week-2 milestone: collect → mine → break → detect."""
    os.environ["ANTHROPIC_API_KEY"] = ""
    baseline = generate_traces(180, seed=11)
    result = await mine_invariants(
        agent_id=uuid.uuid4(), traces=baseline, session=None, persist=False
    )
    invariants = [
        InvariantSpec(
            id=inv.invariant_id,
            type=inv.type,
            description=inv.description,
            formal_rule=inv.formal_rule,
            severity=inv.severity,
            params=inv.params,
        )
        for inv in result.invariants
        if inv.params  # only those checkers can evaluate
    ]
    assert invariants, "demo agent should produce at least one params-bearing invariant"

    # Pick an ordering invariant we expect to be mined consistently.
    # validate_input runs first in every trace, so it's the tool_a position
    # of multiple ordering rules. Rename it in the candidate so the mined
    # ordering "validate_input before <tool_b>" fails on candidate traces.
    ordering_invs = [
        i
        for i in invariants
        if i.type == InvariantType.ORDERING
        and i.params.get("tool_a") == "validate_input"
    ]
    assert ordering_invs, "expected at least one validate_input ordering invariant"

    candidate: list[NormalizedTrace] = []
    for trace in generate_traces(180, seed=11):
        new_steps = []
        for s in trace.steps:
            if isinstance(s, ToolCallStep) and s.tool == "validate_input":
                new_steps.append(s.model_copy(update={"tool": "skip_validate"}))
            else:
                new_steps.append(s)
        candidate.append(trace.model_copy(update={"steps": new_steps}))

    report = compare_trace_sets(invariants, baseline, candidate)
    descriptions = " ".join(f.description for f in report.failures)
    assert "validate_input" in descriptions, (
        f"expected a failure mentioning validate_input, got: {descriptions}"
    )
