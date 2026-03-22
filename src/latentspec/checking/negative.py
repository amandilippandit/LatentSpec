"""Negative invariant checker (§3.3).

Two modes:
  - `forbidden_patterns`: deny-list of tool name substrings.
  - `allowed_repertoire`: closed-world allowlist; any tool not in this
    set is a violation.
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


class NegativeChecker(Checker):
    invariant_type = InvariantType.NEGATIVE

    def check(
        self, invariant: InvariantSpec, trace: NormalizedTrace
    ) -> CheckResult:
        patterns = invariant.params.get("forbidden_patterns") or []
        allowed_repertoire = invariant.params.get("allowed_repertoire") or []

        if not patterns and not allowed_repertoire:
            raise CheckerError(
                f"negative invariant {invariant.id} requires either "
                f"forbidden_patterns or allowed_repertoire"
            )

        # Closed-world repertoire: any tool outside the allowed set violates.
        if allowed_repertoire:
            allowed_set = {str(t) for t in allowed_repertoire}
            for idx, step in enumerate(trace.steps):
                if not isinstance(step, ToolCallStep):
                    continue
                if step.tool not in allowed_set:
                    return self._result(
                        invariant,
                        trace,
                        CheckOutcome.FAIL,
                        ViolationDetails(
                            expected=(
                                f"agent only uses tools in the closed-world allowlist "
                                f"(|repertoire|={len(allowed_set)})"
                            ),
                            observed=(
                                f"agent invoked `{step.tool}` at step {idx} — "
                                f"not in the allowlist"
                            ),
                            affected_step_indices=[idx],
                            extra={
                                "tool": step.tool,
                                "closed_world": True,
                                "category": invariant.params.get("category"),
                            },
                        ),
                    )
            return self._result(invariant, trace, CheckOutcome.PASS)

        # Forbidden-pattern (deny-list) mode.
        patterns_lower = [str(p).lower() for p in patterns]
        for idx, step in enumerate(trace.steps):
            if not isinstance(step, ToolCallStep):
                continue
            tool_lc = step.tool.lower()
            for pattern in patterns_lower:
                if pattern in tool_lc:
                    return self._result(
                        invariant,
                        trace,
                        CheckOutcome.FAIL,
                        ViolationDetails(
                            expected=f"agent never invokes a `{patterns}` action",
                            observed=f"agent called `{step.tool}` at step {idx}",
                            affected_step_indices=[idx],
                            extra={
                                "matched_pattern": pattern,
                                "tool": step.tool,
                                "category": invariant.params.get("category"),
                            },
                        ),
                    )

        return self._result(invariant, trace, CheckOutcome.PASS)
