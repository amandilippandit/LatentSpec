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

    @guardrail(rules, fail_on="critical")
    def delete_user(user_id: str) -> None:
        return None

    with pytest.raises(GuardrailViolation):
        with guarded_turn(rules, user_input="please delete user 42"):
            delete_user("42")


def test_guardrail_fail_on_warn_does_not_raise() -> None:
    rules = RuleSet.from_invariants("agent-1", [_ordering_rule(severity=Severity.MEDIUM)])

    @guardrail(rules, fail_on="critical")  # critical-only — MEDIUM rule shouldn't block
    def send_email(to: str) -> None:
        return None

    with guarded_turn(rules, user_input="email me"):
        # Should not raise even though the rule is violated, because severity is MEDIUM
        send_email("user@example.com")


def test_guardrail_thread_isolation() -> None:
    rules = RuleSet.from_invariants("agent-1", [_ordering_rule()])

    @guardrail(rules, fail_on="critical")
    def send_email(to: str) -> None:
        return None

    @guardrail(rules, fail_on="critical")
    def auth() -> None:
        return None

    errors: list[BaseException] = []

    def run_t1() -> None:
        try:
            with guarded_turn(rules, user_input="t1"):
                auth()
                send_email("a@x.com")
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    def run_t2() -> None:
        try:
            with guarded_turn(rules, user_input="t2"):
                # No auth -> should violate
                send_email("b@x.com")
        except GuardrailViolation:
            return
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
        else:
            errors.append(AssertionError("t2 should have raised"))

    th1 = threading.Thread(target=run_t1)
    th2 = threading.Thread(target=run_t2)
    th1.start(); th2.start()
    th1.join(); th2.join()
    assert errors == [], errors


def test_ruleset_filter_by_severity() -> None:
    rules = RuleSet.from_invariants(
        "agent-1",
        [
            _ordering_rule(severity=Severity.CRITICAL),
            _ordering_rule(severity=Severity.MEDIUM),
        ],
    )
    high_only = rules.filter(min_severity=Severity.HIGH)
    assert len(high_only) == 1
    assert high_only[0].severity == Severity.CRITICAL
