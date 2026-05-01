"""Statistical invariant checker (§3.3).

Two metrics today:
  - `latency_ms`: per-tool p99 latency threshold; flag any step exceeding it.
  - `success_rate`: per-tool success-rate floor; flag any error step against
    a tool that should be reliable.

Latency violations are returned as WARN (rather than FAIL) — they signal
performance regression, not correctness loss. Severity is preserved on the
result so the PR-comment renderer can still surface them prominently.
"""

from __future__ import annotations

from latentspec.checking.base import (
    Checker,
    CheckerError,
    CheckOutcome,
    CheckResult,
    InvariantSpec,
    ViolationDetails,
)
from latentspec.models.invariant import InvariantType
from latentspec.schemas.trace import NormalizedTrace, ToolCallStep


class StatisticalChecker(Checker):
    invariant_type = InvariantType.STATISTICAL

    def check(
        self, invariant: InvariantSpec, trace: NormalizedTrace
    ) -> CheckResult:
        metric = invariant.params.get("metric")
        tool = invariant.params.get("tool")
        if not metric or not tool:
            raise CheckerError(
                f"statistical invariant {invariant.id} missing metric/tool"
            )

        steps = [
            (idx, s)
            for idx, s in enumerate(trace.steps)
            if isinstance(s, ToolCallStep) and s.tool == tool
        ]
        if not steps:
            return self._result(invariant, trace, CheckOutcome.NOT_APPLICABLE)

        if metric == "latency_ms":
            threshold = float(invariant.params.get("threshold") or 0.0)
            offenders = [
                (idx, s)
                for idx, s in steps
                if s.latency_ms is not None and s.latency_ms > threshold
            ]
            if not offenders:
                return self._result(invariant, trace, CheckOutcome.PASS)
            worst_idx, worst_step = max(
                offenders, key=lambda t: t[1].latency_ms or 0
            )
            return self._result(
                invariant,
                trace,
                CheckOutcome.WARN,
                ViolationDetails(
                    expected=f"latency <= {threshold:.0f}ms (p99)",
                    observed=(
                        f"`{tool}` ran in {worst_step.latency_ms}ms at step {worst_idx}"
                    ),
                    affected_step_indices=[idx for idx, _ in offenders],
                    metric="latency_ms",
                    threshold=threshold,
                    actual=float(worst_step.latency_ms or 0.0),
                    extra={"tool": tool, "offender_count": len(offenders)},
                ),
            )

        if metric == "success_rate":
            min_rate = float(invariant.params.get("rate") or 0.95)
            errors = [
                (idx, s)
                for idx, s in steps
                if (s.result_status or "success") != "success"
            ]
            success_rate = 1.0 - (len(errors) / max(1, len(steps)))
            if success_rate >= min_rate:
                return self._result(invariant, trace, CheckOutcome.PASS)
            return self._result(
                invariant,
                trace,
                CheckOutcome.FAIL,
                ViolationDetails(
                    expected=f"success rate >= {min_rate:.2%}",
                    observed=f"`{tool}` succeeded at {success_rate:.2%}",
                    affected_step_indices=[idx for idx, _ in errors],
                    metric="success_rate",
                    threshold=min_rate,
                    actual=round(success_rate, 4),
                    extra={"tool": tool, "errors_in_trace": len(errors)},
                ),
            )

        raise CheckerError(f"unknown statistical metric: {metric!r}")
