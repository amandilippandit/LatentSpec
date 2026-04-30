"""Tests for the Z3 SMT compiler + verifier (§3.2 / §10.1)."""

from __future__ import annotations

from datetime import UTC, datetime

from latentspec.models.invariant import InvariantType
from latentspec.schemas.trace import (
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)
from latentspec.smt.certificates import generate_certificate
from latentspec.smt.compiler import Z3CompilerError, compile_invariant
from latentspec.smt.verifier import verify_trace


def _trace(steps, *, segment: str | None = None) -> NormalizedTrace:
    return NormalizedTrace(
        trace_id="t",
        agent_id="a",
        timestamp=datetime.now(UTC),
        steps=list(steps),
        metadata=TraceMetadata(user_segment=segment),
    )


def test_z3_ordering_holds_for_correct_trace() -> None:
    comp = compile_invariant(
        InvariantType.ORDERING, {"tool_a": "auth", "tool_b": "db_write"}
    )
    good = _trace(
        [
            UserInputStep(content="hi"),
            ToolCallStep(tool="auth", args={}),
            ToolCallStep(tool="db_write", args={}),
        ]
    )
    result = verify_trace(comp, good, timeout_ms=2000)
    assert result.holds, result


def test_z3_ordering_fails_with_counter_example() -> None:
    comp = compile_invariant(
        InvariantType.ORDERING, {"tool_a": "auth", "tool_b": "db_write"}
    )
    bad = _trace(
        [
            UserInputStep(content="hi"),
            ToolCallStep(tool="db_write", args={}),
            ToolCallStep(tool="auth", args={}),
        ]
    )
    result = verify_trace(comp, bad, timeout_ms=2000)
    assert not result.holds
    assert result.counter_example is not None
    assert result.counter_example.get("n") == 3


def test_z3_negative_holds_when_dangerous_absent() -> None:
    comp = compile_invariant(
        InvariantType.NEGATIVE, {"forbidden_patterns": ["delete", "drop"]}
    )
    benign = _trace(
        [
            UserInputStep(content="hi"),
            ToolCallStep(tool="search_flights", args={}),
            ToolCallStep(tool="book_flight", args={}),
        ]
    )
    assert verify_trace(comp, benign, timeout_ms=2000).holds


def test_z3_negative_fails_when_forbidden_present() -> None:
    comp = compile_invariant(
        InvariantType.NEGATIVE, {"forbidden_patterns": ["delete"]}
    )
    bad = _trace(
        [
            UserInputStep(content="hi"),
            ToolCallStep(tool="delete", args={}),
        ]
    )
    assert not verify_trace(comp, bad, timeout_ms=2000).holds


def test_z3_statistical_latency_holds_for_low_latency() -> None:
    comp = compile_invariant(
        InvariantType.STATISTICAL,
        {"metric": "latency_ms", "tool": "search", "threshold": 500},
    )
    fast = _trace(
        [
            ToolCallStep(tool="search", args={}, latency_ms=200),
            ToolCallStep(tool="search", args={}, latency_ms=300),
        ]
    )
    slow = _trace(
        [
            ToolCallStep(tool="search", args={}, latency_ms=2000),
        ]
    )
    assert verify_trace(comp, fast, timeout_ms=2000).holds
    assert not verify_trace(comp, slow, timeout_ms=2000).holds


def test_z3_state_invariant_after_terminator() -> None:
    comp = compile_invariant(
        InvariantType.STATE,
        {
            "terminator_tool": "session_close",
            "forbidden_after": ["read_user_data", "write_user_data"],
        },
    )
    ok = _trace(
        [
            ToolCallStep(tool="read_user_data", args={}),
            ToolCallStep(tool="session_close", args={}),
        ]
    )
    bad = _trace(
        [
            ToolCallStep(tool="session_close", args={}),
            ToolCallStep(tool="read_user_data", args={}),
        ]
    )
    assert verify_trace(comp, ok, timeout_ms=2000).holds
    assert not verify_trace(comp, bad, timeout_ms=2000).holds


def test_z3_compiler_rejects_bad_params() -> None:
    try:
        compile_invariant(InvariantType.ORDERING, {})
    except Z3CompilerError:
        return
    raise AssertionError("expected Z3CompilerError on missing params")


def test_certificate_generates_signature_when_key_set(monkeypatch) -> None:
    monkeypatch.setenv("LATENTSPEC_CERT_SIGNING_KEY", "test-secret-1234567890")
    comp = compile_invariant(
        InvariantType.NEGATIVE, {"forbidden_patterns": ["delete"]}
    )
    sample = [
        _trace(
            [
                ToolCallStep(tool="search_flights", args={}),
                ToolCallStep(tool="book_flight", args={}),
            ]
        )
        for _ in range(3)
    ]
    cert = generate_certificate(comp, sample, mode="combined", timeout_ms_per_trace=1000)
    assert cert.empirical is not None
    assert cert.empirical.sample_size == 3
    assert cert.empirical.sample_holds == 3
    assert cert.empirical.sample_violates == 0
    assert cert.signature_hex is not None
    assert len(cert.signature_hex) == 64  # SHA-256 hex
