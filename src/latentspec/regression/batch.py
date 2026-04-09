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
    baseline: Iterable[NormalizedTrace],
    candidate: Iterable[NormalizedTrace],
    *,
    regression_drop: float = _REGRESSION_DROP,
    min_applicable: int = _MIN_APPLICABLE,
) -> RegressionReport:
    """Score baseline & candidate on the same invariant set; flag regressions.

    A FAILURE is reported when:
      - the candidate fail_rate strictly exceeds the baseline fail_rate, AND
      - the candidate has at least `min_applicable` applicable traces, AND
      - the candidate pass_rate dropped by at least `regression_drop`.

    A WARNING (e.g. perf regression) is reported when warn_rate goes up
    by at least `regression_drop`, even without a fail-rate change.
    """
    invariants = list(invariants)
    baseline_traces = list(baseline)
    candidate_traces = list(candidate)

    baseline_results = check_traces(invariants, baseline_traces)
    candidate_results = check_traces(invariants, candidate_traces)

    bl_by_inv: dict[str, list[CheckResult]] = defaultdict(list)
    cd_by_inv: dict[str, list[CheckResult]] = defaultdict(list)
    for r in baseline_results:
        bl_by_inv[r.invariant_id].append(r)
    for r in candidate_results:
        cd_by_inv[r.invariant_id].append(r)

    baseline_summaries: list[InvariantBatchSummary] = []
    candidate_summaries: list[InvariantBatchSummary] = []
    failures: list[InvariantBatchSummary] = []
    warnings: list[InvariantBatchSummary] = []
    passes = 0

    for inv in invariants:
        bl = _summarize(inv, bl_by_inv.get(inv.id, []))
        cd = _summarize(inv, cd_by_inv.get(inv.id, []))
        baseline_summaries.append(bl)
        candidate_summaries.append(cd)

        if cd.applicable < min_applicable:
            passes += 1
            continue

        # FAIL: candidate fail_rate is materially higher than baseline.
        if (
            cd.fail_rate > bl.fail_rate
            and cd.pass_rate <= bl.pass_rate - regression_drop
        ):
            failures.append(cd)
            continue

        # WARN: perf-style regression (warn_rate up).
        if cd.warn_rate >= bl.warn_rate + regression_drop:
            warnings.append(cd)
            continue

        passes += 1

    counts = Counter(
        {
            "PASS": passes,
            "WARN": len(warnings),
            "FAIL": len(failures),
            "CHECKED": len(invariants),
        }
    )

    return RegressionReport(
        invariants_checked=len(invariants),
        baseline=baseline_summaries,
        candidate=candidate_summaries,
        failures=failures,
        warnings=warnings,
        passes=passes,
        counts=dict(counts),
    )


def _exit_code_for(report: RegressionReport, fail_on: str) -> int:
    """Compute the §4.2 exit code from a regression report.

    fail_on:
      - "critical": exit non-zero only on critical-severity failures
      - "high":     non-zero on critical or high failures
      - "any":      non-zero on any failure
      - "warn":     non-zero on any warning or failure
      - "never":    always exit 0 (advisory mode)
    """
    fail_on = fail_on.lower()
    if fail_on == "never":
        return 0
    if fail_on == "warn":
        if report.failures or report.warnings:
            return 1
        return 0
    if fail_on == "any":
        return 1 if report.failures else 0
    if fail_on == "high":
        critical = [f for f in report.failures if f.severity in (Severity.CRITICAL, Severity.HIGH)]
        return 1 if critical else 0
    # default: critical-only
    critical = [f for f in report.failures if f.severity == Severity.CRITICAL]
    return 1 if critical else 0
