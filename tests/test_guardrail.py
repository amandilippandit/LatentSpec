"""Tests for the inline guardrail SDK (`@guardrail`, `guarded_turn`)."""

from __future__ import annotations

import threading

import pytest

from latentspec.checking.base import InvariantSpec
from latentspec.models.invariant import InvariantType, Severity
from latentspec.sdk.guardrail import (
    GuardrailViolation,
    RuleSet,
    guarded_turn,
    guardrail,
)


def _ordering_rule(*, severity: Severity = Severity.CRITICAL) -> InvariantSpec:
    return InvariantSpec(
        id="inv-1",
        type=InvariantType.ORDERING,
        description="`auth` must precede `send_email`",
        formal_rule="placeholder",
        severity=severity,
        params={"tool_a": "auth", "tool_b": "send_email"},
    )


def _negative_rule() -> InvariantSpec:
    return InvariantSpec(
        id="inv-2",
        type=InvariantType.NEGATIVE,
        description="never call `delete_user`",
        formal_rule="placeholder",
        severity=Severity.CRITICAL,
        params={"forbidden_patterns": ["delete_user"], "category": "delete"},
    )


def test_guardrail_blocks_when_precondition_missing() -> None:
    rules = RuleSet.from_invariants("agent-1", [_ordering_rule()])

    @guardrail(rules, fail_on="critical")
    def send_email(to: str) -> str:
        return f"sent {to}"

    with guarded_turn(rules, user_input="email me"):
        with pytest.raises(GuardrailViolation) as ei:
            send_email("user@example.com")
    assert ei.value.invariant_id == "inv-1"
    assert ei.value.severity == Severity.CRITICAL


def test_guardrail_allows_after_satisfying_precondition() -> None:
    rules = RuleSet.from_invariants("agent-1", [_ordering_rule()])

    @guardrail(rules, fail_on="critical")
    def auth() -> None:
        return None

    @guardrail(rules, fail_on="critical")
    def send_email(to: str) -> str:
        return f"sent {to}"

    with guarded_turn(rules, user_input="email me"):
        auth()
        assert send_email("user@example.com") == "sent user@example.com"


def test_guardrail_blocks_negative_violation() -> None:
    rules = RuleSet.from_invariants("agent-1", [_negative_rule()])
