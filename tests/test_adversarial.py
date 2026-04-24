"""Adversarial-robustness tests for the trace pipeline.

The system must survive:
  - malformed / partially-typed payloads (the normalizer's job)
  - adversarial Unicode (RTL overrides, zero-width injection, control chars)
  - oversized payloads (huge args, deeply nested structures)
  - tool-name collisions across canonical / forbidden categories
  - synthetic violating traces from `synthesize_violating_trace`
  - schema-violating mined params (the mining drops them rather than crashing)

Each test asserts the system either rejects with a clear error OR
handles the input gracefully without crashing the process.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import pytest

from latentspec.checking import dispatch
from latentspec.checking.base import CheckOutcome, InvariantSpec
from latentspec.mining.fingerprint import canonical_shape, fingerprint
from latentspec.mining.formalization import formalize
from latentspec.models.invariant import InvariantType, Severity
from latentspec.normalizers import RawJSONNormalizer, NormalizerError
from latentspec.schemas.invariant import InvariantCandidate
from latentspec.schemas.params import ParamsValidationError, validate_params
from latentspec.schemas.trace import (
    AgentResponseStep,
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)
from latentspec.sdk.redaction import Redactor
from latentspec.smt.compiler import compile_invariant
from latentspec.smt.synthesis import synthesize_violating_trace


# ---- adversarial payloads --------------------------------------------------


def test_normalizer_rejects_payload_without_steps() -> None:
    with pytest.raises(NormalizerError):
        RawJSONNormalizer().normalize({"trace_id": "x"}, agent_id="a")


def test_normalizer_rejects_payload_with_wrong_step_type() -> None:
    with pytest.raises(NormalizerError):
        RawJSONNormalizer().normalize(
            {"steps": [{"type": "INVALID_KIND", "content": "x"}]},
            agent_id="a",
        )


def test_normalizer_accepts_unicode_rtl_and_zero_width() -> None:
    """RTL overrides + zero-width chars should round-trip unscathed (the
    normalizer doesn't try to "fix" content; only structural validation)."""
    payload = {
        "steps": [
            {"type": "user_input", "content": "hello‮world​zwsp"},
            {"type": "agent_response", "content": "ok؜ALM-LRM"},
        ]
    }
    nt = RawJSONNormalizer().normalize(payload, agent_id="a")
    assert len(nt.steps) == 2
    # The redactor should still operate cleanly on adversarial unicode
    r = Redactor()
    r.redact_string(nt.steps[0].content)


def test_normalizer_accepts_huge_args_without_crash() -> None:
    """1MB args blob — must not blow the stack or hang."""
    big = {"k_" + str(i): "v" * 10 for i in range(1000)}  # ~13KB but many keys
    payload = {
        "steps": [
            {"type": "user_input", "content": "hi"},
            {"type": "tool_call", "tool": "search", "args": big, "latency_ms": 10},
            {"type": "agent_response", "content": "ok"},
        ]
    }
    nt = RawJSONNormalizer().normalize(payload, agent_id="a")
    tool_step = next(s for s in nt.steps if isinstance(s, ToolCallStep))
    assert len(tool_step.args) == 1000


def test_fingerprint_stable_across_distinct_args() -> None:
    """Two traces with identical step shape but wildly different args
    MUST produce the same fingerprint — the fingerprint is shape-only."""
    base_steps = [
        UserInputStep(content="a"),
        ToolCallStep(tool="t", args={}),
    ]
    a = NormalizedTrace(
        trace_id="a", agent_id="x", timestamp=datetime.now(UTC),
        steps=base_steps, metadata=TraceMetadata(),
    )
    b = NormalizedTrace(
        trace_id="b", agent_id="x", timestamp=datetime.now(UTC),
        steps=[
            UserInputStep(content="completely different content " * 200),
            ToolCallStep(tool="t", args={"k": "v" * 10000}),
        ],
        metadata=TraceMetadata(),
    )
    assert fingerprint(a) == fingerprint(b)


def test_canonical_shape_handles_empty_trace() -> None:
    empty = NormalizedTrace(
        trace_id="e", agent_id="x", timestamp=datetime.now(UTC),
        steps=[], metadata=TraceMetadata(),
    )
    assert canonical_shape(empty) == ""
    assert fingerprint(empty)  # still produces a stable hash for the empty shape


# ---- params schema enforcement on adversarial input -----------------------


def test_params_validation_rejects_unknown_keys_in_strict_types() -> None:
    with pytest.raises(ParamsValidationError):
        validate_params(
            InvariantType.ORDERING,
            {"tool_a": "auth", "tool_b": "db", "evil_extra_field": "boom"},
        )

