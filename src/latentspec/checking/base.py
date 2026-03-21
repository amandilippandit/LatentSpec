"""Checker protocol and result types.

A checker takes one normalized trace + one invariant and returns a
CheckResult — pass / fail / warn / not_applicable. Failures carry a
`ViolationDetails` payload that the §4.2 PR-comment renderer + §4.3
root-cause analyzer both read from.

`InvariantSpec` is a transport-agnostic snapshot of an invariant suitable
for both DB-backed (`Invariant` ORM) and JSON-loaded (CLI from-file) flows.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from latentspec.models.invariant import InvariantType, Severity
from latentspec.schemas.trace import NormalizedTrace


class CheckerError(Exception):
    """Raised when a check cannot be evaluated (malformed rule, missing params)."""


class CheckOutcome(str, enum.Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class InvariantSpec:
    """Minimal invariant view a checker needs."""

    id: str
    type: InvariantType
    description: str
    formal_rule: str
    severity: Severity
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ViolationDetails:
    """§4.3 violation analysis bundle attached to a failed CheckResult."""

    expected: str
    observed: str
    affected_step_indices: list[int] = field(default_factory=list)
    metric: str | None = None
    threshold: float | None = None
    actual: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckResult:
    invariant_id: str
    invariant_type: InvariantType
    invariant_description: str
    severity: Severity
    trace_id: str
    outcome: CheckOutcome
    details: ViolationDetails | None = None
    duration_ms: float = 0.0


@runtime_checkable
class CheckerProto(Protocol):
    """Structural typing for checker callables."""

    def __call__(
        self, invariant: InvariantSpec, trace: NormalizedTrace
    ) -> CheckResult: ...


class Checker(ABC):
    """Inherit from this to implement a per-type rule evaluator."""

    invariant_type: InvariantType

    @abstractmethod
    def check(
        self, invariant: InvariantSpec, trace: NormalizedTrace
    ) -> CheckResult:
        ...

    def _result(
        self,
        invariant: InvariantSpec,
        trace: NormalizedTrace,
        outcome: CheckOutcome,
        details: ViolationDetails | None = None,
    ) -> CheckResult:
        return CheckResult(
            invariant_id=invariant.id,
            invariant_type=invariant.type,
            invariant_description=invariant.description,
            severity=invariant.severity,
            trace_id=trace.trace_id,
            outcome=outcome,
            details=details,
        )
