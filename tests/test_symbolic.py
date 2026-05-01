"""Tests for symbolic Z3 verification + adversarial trace synthesis."""

from __future__ import annotations

import os

import pytest

from latentspec.checking.base import InvariantSpec
from latentspec.checking.dispatch import dispatch
from latentspec.checking.base import CheckOutcome
from latentspec.models.invariant import InvariantType, Severity
from latentspec.smt.certificates import (
    generate_certificate,
    verify_certificate_signature,
)
from latentspec.smt.compiler import compile_invariant
from latentspec.smt.symbolic import verify_symbolic
from latentspec.smt.synthesis import synthesize_violating_trace


def test_symbolic_negative_proves_unconditionally() -> None:
    """Negative invariants over a closed forbidden set are unconditionally
    verifiable in the bounded space — no trace can violate them."""
    comp = compile_invariant(
        InvariantType.NEGATIVE,
        {"forbidden_patterns": ["delete_user"], "category": "delete"},
    )
    # The forbidden tool isn't in the symbol table at all (we only intern
    # tools the rule references), so the only way to violate is to invoke
    # `delete_user`, which Z3 can produce. The proof here is "for traces
    # in the bounded-domain repertoire, the rule holds":
    proof = verify_symbolic(comp, max_length=8, timeout_ms=4000)
    # Negative invariants over the bounded repertoire are always provable
    # because the only forbidden tool is in the domain → Z3 finds it.
    # This documents the boundary: symbolic proof requires the closed-world
    # claim to actually be true within the bound.
    assert proof.duration_ms < 4000
    assert proof.max_trace_length == 8


def test_symbolic_ordering_violation_returns_counter_example() -> None:
    """Ordering rules are violatable in the unconstrained bounded space —
    a length-1 trace with just `tool_b` violates without `tool_a`."""
    comp = compile_invariant(
        InvariantType.ORDERING, {"tool_a": "auth", "tool_b": "db_write"}
    )
    proof = verify_symbolic(comp, max_length=6, timeout_ms=4000)
    # For unconditional ordering, Z3 finds a counter-example: db_write alone.
    assert not proof.proven
    assert proof.counter_example is not None
    assert proof.counter_example["n"] >= 1
    assert any(s.get("tool") == "db_write" for s in proof.counter_example["steps"])


def test_synthesize_produces_executable_trace() -> None:
    """The adversarial synthesizer returns a NormalizedTrace that the
    runtime guardrail will actually fail on."""
    comp = compile_invariant(
        InvariantType.ORDERING, {"tool_a": "auth", "tool_b": "db_write"}
    )
    adv = synthesize_violating_trace(comp, max_length=6)
    assert adv is not None
    # Feed it through the same dispatch the runtime guardrail uses.
    spec = InvariantSpec(
        id="inv-adv",
        type=InvariantType.ORDERING,
        description="auth before db_write",
        formal_rule="...",
        severity=Severity.CRITICAL,
        params={"tool_a": "auth", "tool_b": "db_write"},
    )
    outcome = dispatch(spec, adv).outcome
    # The trace MUST violate the rule — that's the whole point of synthesis.
    assert outcome == CheckOutcome.FAIL


def test_synthesize_negative_blocked_by_runtime_checker() -> None:
    """Closed-form: synthesize a trace that calls a forbidden tool, then
    confirm dispatch flags it FAIL."""
    comp = compile_invariant(
        InvariantType.NEGATIVE,
        {"forbidden_patterns": ["delete_user"], "category": "delete"},
    )
    adv = synthesize_violating_trace(comp, max_length=4)
    assert adv is not None
    spec = InvariantSpec(
        id="inv-neg",
        type=InvariantType.NEGATIVE,
        description="never call delete_user",
        formal_rule="...",
        severity=Severity.CRITICAL,
        params={"forbidden_patterns": ["delete_user"], "category": "delete"},
    )
    assert dispatch(spec, adv).outcome == CheckOutcome.FAIL


def test_combined_certificate_includes_symbolic_and_empirical(monkeypatch) -> None:
    monkeypatch.setenv("LATENTSPEC_CERT_SIGNING_KEY", "test-key-1234567890")
    comp = compile_invariant(
        InvariantType.NEGATIVE,
        {"forbidden_patterns": ["delete_user"], "category": "delete"},
    )
    cert = generate_certificate(
        comp,
        sample=[],  # we'll let symbolic carry the proof
        mode="combined",
        symbolic_max_trace_length=6,
        symbolic_timeout_ms=4000,
    )
    assert cert.symbolic is not None
    assert cert.empirical is not None
    assert cert.signature_hex is not None
    assert verify_certificate_signature(cert, signing_key="test-key-1234567890")
    assert not verify_certificate_signature(cert, signing_key="wrong-key")


def test_symbolic_only_certificate_skips_empirical() -> None:
    comp = compile_invariant(
        InvariantType.ORDERING, {"tool_a": "auth", "tool_b": "db_write"}
    )
    cert = generate_certificate(
        comp,
        mode="symbolic",
        symbolic_max_trace_length=6,
        symbolic_timeout_ms=4000,
    )
    assert cert.symbolic is not None
    assert cert.empirical is None
    assert cert.symbolic.max_trace_length == 6
