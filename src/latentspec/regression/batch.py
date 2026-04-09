"""§4.1 batch comparison mode.

Run every invariant against the baseline set and the candidate set, then
surface invariants whose pass-rate dropped meaningfully in the candidate.
The unit of comparison is per-invariant pass-rate (over traces where the
rule applied), not per-trace verdict — a single bad trace shouldn't fail a
build, a real regression in the population should.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from latentspec.checking.base import (
    CheckOutcome,
    CheckResult,
    InvariantSpec,
)
from latentspec.checking.runner import check_traces
from latentspec.models.invariant import InvariantType, Severity
from latentspec.schemas.trace import NormalizedTrace


@dataclass
class InvariantBatchSummary:
    invariant_id: str
    type: InvariantType
    description: str
    severity: Severity
    applicable: int = 0
    passed: int = 0
    failed: int = 0
    warned: int = 0
    pass_rate: float = 1.0
    fail_rate: float = 0.0
    warn_rate: float = 0.0
    sample_failure_traces: list[str] = field(default_factory=list)
    sample_warn_traces: list[str] = field(default_factory=list)


def _summarize(
    invariant: InvariantSpec, results: list[CheckResult]
) -> InvariantBatchSummary:
    summary = InvariantBatchSummary(
        invariant_id=invariant.id,
        type=invariant.type,
        description=invariant.description,
        severity=invariant.severity,
    )
    for r in results:
        if r.outcome == CheckOutcome.NOT_APPLICABLE:
            continue
        summary.applicable += 1
        if r.outcome == CheckOutcome.PASS:
            summary.passed += 1
        elif r.outcome == CheckOutcome.FAIL:
            summary.failed += 1
            if len(summary.sample_failure_traces) < 5:
                summary.sample_failure_traces.append(r.trace_id)
        elif r.outcome == CheckOutcome.WARN:
            summary.warned += 1
            if len(summary.sample_warn_traces) < 5:
                summary.sample_warn_traces.append(r.trace_id)

    n = max(1, summary.applicable)
    summary.pass_rate = round(summary.passed / n, 4)
    summary.fail_rate = round(summary.failed / n, 4)
    summary.warn_rate = round(summary.warned / n, 4)
    return summary


@dataclass
class RegressionReport:
    """Output of comparing baseline → candidate."""

    invariants_checked: int
    baseline: list[InvariantBatchSummary]
    candidate: list[InvariantBatchSummary]
    failures: list[InvariantBatchSummary] = field(default_factory=list)
    warnings: list[InvariantBatchSummary] = field(default_factory=list)
    passes: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def has_critical_failures(self) -> bool:
        return any(
            f.severity == Severity.CRITICAL for f in self.failures
        ) or any(
            f.severity == Severity.HIGH for f in self.failures
        )

    @property
    def exit_code_for(self, fail_on: str = "critical") -> int:  # type: ignore[override]
        return _exit_code_for(self, fail_on)


# Min applicable traces below which we won't trust pass-rate movement
_MIN_APPLICABLE = 3
# Pass-rate drop large enough to flag as a regression (10pp by default)
_REGRESSION_DROP = 0.10


def compare_trace_sets(
    invariants: Iterable[InvariantSpec],
