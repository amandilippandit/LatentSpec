"""Conditional invariant checker (§3.3).

If user input contains `keyword`, the agent must invoke `tool`. Returns
NOT_APPLICABLE when the keyword isn't present, PASS when it is and the tool
runs, FAIL otherwise.
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
from latentspec.schemas.trace import (
    NormalizedTrace,
    ToolCallStep,
    UserInputStep,
)


class ConditionalChecker(Checker):
    invariant_type = InvariantType.CONDITIONAL

    def check(
        self, invariant: InvariantSpec, trace: NormalizedTrace
    ) -> CheckResult:
        keyword = (invariant.params.get("keyword") or "").lower()
        tool = invariant.params.get("tool")
        if not keyword or not tool:
            raise CheckerError(
                f"conditional invariant {invariant.id} missing keyword/tool in params"
            )

        user_text = " ".join(
            s.content for s in trace.steps if isinstance(s, UserInputStep)
        ).lower()
        if keyword not in user_text:
            return self._result(invariant, trace, CheckOutcome.NOT_APPLICABLE)

        tool_indices = [
            idx
            for idx, s in enumerate(trace.steps)
            if isinstance(s, ToolCallStep) and s.tool == tool
        ]
        if tool_indices:
            return self._result(invariant, trace, CheckOutcome.PASS)

        return self._result(
            invariant,
            trace,
            CheckOutcome.FAIL,
            ViolationDetails(
                expected=f"tool `{tool}` invoked when input mentions '{keyword}'",
                observed=(
                    f"input mentioned '{keyword}' but `{tool}` was never called"
                ),
                extra={"keyword": keyword, "tool": tool},
            ),
        )
