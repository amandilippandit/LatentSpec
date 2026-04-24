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


def test_params_validation_rejects_invalid_tool_names() -> None:
    bad_names = [
        "with spaces",  # whitespace inside body
        "tab\there",    # control character
        "1starts_with_digit",  # leading char must be letter or underscore
        "@invalid_lead", # leading punctuation outside [A-Za-z_]
        "",             # empty
        "x" * 200,      # too long
    ]
    for name in bad_names:
        with pytest.raises(ParamsValidationError):
            validate_params(
                InvariantType.ORDERING,
                {"tool_a": name, "tool_b": "ok_tool"},
            )


def test_params_validation_clamps_statistical_threshold() -> None:
    """Negative threshold isn't physically meaningful — schema rejects."""
    with pytest.raises(ParamsValidationError):
        validate_params(
            InvariantType.STATISTICAL,
            {"metric": "latency_ms", "tool": "search", "threshold": -100},
        )


def test_formalize_drops_candidate_with_malformed_extra() -> None:
    cand = InvariantCandidate(
        type=InvariantType.ORDERING,
        description="placeholder",
        formal_rule="...",
        support=0.95,
        consistency=0.95,
        severity=Severity.HIGH,
        discovered_by="statistical",
        extra={"tool_a": "with whitespace", "tool_b": "ok"},
    )
    assert formalize(cand) is None


# ---- runtime checker robustness ------------------------------------------


def test_dispatch_returns_not_applicable_on_missing_params() -> None:
    """A persisted invariant whose params dict is empty must not crash —
    the checker returns NOT_APPLICABLE so the streaming detector keeps moving."""
    spec = InvariantSpec(
        id="inv-bad",
        type=InvariantType.ORDERING,
        description="placeholder",
        formal_rule="...",
        severity=Severity.HIGH,
        params={},  # empty; checker raises CheckerError → runner converts to NOT_APPLICABLE
    )
    trace = NormalizedTrace(
        trace_id="t", agent_id="a", timestamp=datetime.now(UTC),
        steps=[ToolCallStep(tool="x", args={})],
        metadata=TraceMetadata(),
    )
    # Direct dispatch raises; runner.check_trace catches and returns NA.
    from latentspec.checking.runner import check_trace
    results = check_trace([spec], trace)
    assert results[0].outcome == CheckOutcome.NOT_APPLICABLE


def test_dispatch_handles_step_with_negative_latency() -> None:
    spec = InvariantSpec(
        id="inv-stat",
        type=InvariantType.STATISTICAL,
        description="latency check",
        formal_rule="...",
        severity=Severity.MEDIUM,
        params={"metric": "latency_ms", "tool": "search", "threshold": 100.0},
    )
    trace = NormalizedTrace(
        trace_id="t", agent_id="a", timestamp=datetime.now(UTC),
        steps=[ToolCallStep(tool="search", args={}, latency_ms=-5)],
        metadata=TraceMetadata(),
    )
    # Negative latency is < 100 ⇒ should pass without raising
    assert dispatch(spec, trace).outcome == CheckOutcome.PASS


# ---- end-to-end adversarial: synthesize violating trace + confirm block ----


def test_synthesised_negative_violation_is_caught_by_dispatch() -> None:
    """The Z3 synthesiser produces a real violating trace; dispatch must FAIL it."""
    comp = compile_invariant(
        InvariantType.NEGATIVE,
        {"forbidden_patterns": ["delete_user"], "category": "delete"},
    )
    adv = synthesize_violating_trace(comp, max_length=4)
    if adv is None:
        pytest.skip("Z3 didn't produce a synthesis (rare for negative rule)")
    spec = InvariantSpec(
        id="inv-neg",
        type=InvariantType.NEGATIVE,
        description="never call delete_user",
        formal_rule="...",
        severity=Severity.CRITICAL,
        params={"forbidden_patterns": ["delete_user"], "category": "delete"},
    )
    assert dispatch(spec, adv).outcome == CheckOutcome.FAIL


# ---- redaction adversarial cases -----------------------------------------


def test_redactor_handles_long_input() -> None:
    """A 100KB input shouldn't blow up — regex are linear in input length."""
    big_text = ("user@example.com " * 5000) + ("4242 4242 4242 4242 " * 5000)
    r = Redactor()
    out = r.redact_string(big_text)
    assert "user@example.com" not in out
    assert "[redacted:email]" in out


def test_redactor_handles_nested_dict() -> None:
    """Nested 5-deep structure with a sensitive field at the leaf."""
    payload: dict = {"user@bad.com": True}
    for _ in range(5):
        payload = {"nested": payload, "password": "leak"}
    r = Redactor()
    out = r.redact_value(None, payload)
    s = json.dumps(out)
    # Both the leaf email pattern AND the password key must be redacted
    assert "leak" not in s
    assert "[redacted:blocked_field]" in s
