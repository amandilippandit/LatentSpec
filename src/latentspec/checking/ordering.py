"""Ordering invariant checker (§3.3).

`tool_a` must always appear before `tool_b` within a trace. The checker
returns NOT_APPLICABLE when neither tool is invoked, PASS when only `tool_a`
or neither preceding case applies, and FAIL when `tool_b` is invoked
without a preceding `tool_a`.
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


class OrderingChecker(Checker):
    invariant_type = InvariantType.ORDERING

    def check(
        self, invariant: InvariantSpec, trace: NormalizedTrace
    ) -> CheckResult:
        tool_a = invariant.params.get("tool_a")
        tool_b = invariant.params.get("tool_b")
        if not tool_a or not tool_b:
            raise CheckerError(
                f"ordering invariant {invariant.id} missing tool_a/tool_b in params"
            )

        first_a_index: int | None = None
        first_b_index: int | None = None
        for idx, step in enumerate(trace.steps):
            if not isinstance(step, ToolCallStep):
                continue
            if step.tool == tool_a and first_a_index is None:
                first_a_index = idx
            elif step.tool == tool_b and first_b_index is None:
                first_b_index = idx

        if first_b_index is None:
            return self._result(invariant, trace, CheckOutcome.NOT_APPLICABLE)
        if first_a_index is not None and first_a_index < first_b_index:
            return self._result(invariant, trace, CheckOutcome.PASS)

        return self._result(
            invariant,
            trace,
            CheckOutcome.FAIL,
            ViolationDetails(
                expected=f"`{tool_a}` called before `{tool_b}`",
                observed=(
                    f"`{tool_b}` called at step {first_b_index} "
                    + (
                        f"with no preceding `{tool_a}`"
                        if first_a_index is None
                        else f"but `{tool_a}` came at step {first_a_index}"
                    )
                ),
                affected_step_indices=[first_b_index],
                extra={"tool_a": tool_a, "tool_b": tool_b},
            ),
        )
