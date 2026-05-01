"""State invariant checker (§3.3).

After `terminator_tool` is invoked, the agent must not invoke any tool in
`forbidden_after`. Maps to the doc's example: "never access user data after
session.close()".
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


class StateChecker(Checker):
    invariant_type = InvariantType.STATE

    def check(
        self, invariant: InvariantSpec, trace: NormalizedTrace
    ) -> CheckResult:
        terminator = invariant.params.get("terminator_tool")
        forbidden_after = set(invariant.params.get("forbidden_after") or [])
        if not terminator or not forbidden_after:
            raise CheckerError(
                f"state invariant {invariant.id} needs terminator_tool + forbidden_after"
            )

        terminator_idx: int | None = None
        for idx, step in enumerate(trace.steps):
            if not isinstance(step, ToolCallStep):
                continue
            if terminator_idx is None and step.tool == terminator:
                terminator_idx = idx
                continue
            if terminator_idx is not None and step.tool in forbidden_after:
                return self._result(
                    invariant,
                    trace,
                    CheckOutcome.FAIL,
                    ViolationDetails(
                        expected=(
                            f"no calls to {sorted(forbidden_after)} "
                            f"after `{terminator}`"
                        ),
                        observed=(
                            f"`{step.tool}` invoked at step {idx}, "
                            f"after `{terminator}` at step {terminator_idx}"
                        ),
                        affected_step_indices=[terminator_idx, idx],
                        extra={
                            "terminator_tool": terminator,
                            "violating_tool": step.tool,
                        },
                    ),
                )

        if terminator_idx is None:
            return self._result(invariant, trace, CheckOutcome.NOT_APPLICABLE)
        return self._result(invariant, trace, CheckOutcome.PASS)
