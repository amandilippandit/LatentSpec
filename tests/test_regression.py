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
