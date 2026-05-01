"""Tool selection invariant checker (§3.3).

When the trace's segment matches `segment` (read from
`trace.metadata.user_segment`), the agent must use `expected_tool` and not
the `forbidden_tool`. Maps to the doc's "EU customers → payments_v2,
US → payments_v1" example.
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


class ToolSelectionChecker(Checker):
    invariant_type = InvariantType.TOOL_SELECTION

    def check(
        self, invariant: InvariantSpec, trace: NormalizedTrace
    ) -> CheckResult:
        segment = invariant.params.get("segment")
        expected_tool = invariant.params.get("expected_tool")
        forbidden_tool = invariant.params.get("forbidden_tool")
        if not segment or not expected_tool:
            raise CheckerError(
                f"tool_selection invariant {invariant.id} needs segment + expected_tool"
            )

        trace_segment = (trace.metadata.user_segment or "").lower()
        if trace_segment != segment.lower():
            return self._result(invariant, trace, CheckOutcome.NOT_APPLICABLE)

        expected_indices: list[int] = []
        forbidden_indices: list[int] = []
        for idx, step in enumerate(trace.steps):
            if not isinstance(step, ToolCallStep):
                continue
            if step.tool == expected_tool:
                expected_indices.append(idx)
            elif forbidden_tool and step.tool == forbidden_tool:
                forbidden_indices.append(idx)

        if forbidden_indices:
            return self._result(
                invariant,
                trace,
                CheckOutcome.FAIL,
                ViolationDetails(
                    expected=f"segment {segment!r} routes to `{expected_tool}`",
                    observed=(
                        f"`{forbidden_tool}` invoked at step {forbidden_indices[0]} "
                        f"for {segment!r} segment"
                    ),
                    affected_step_indices=forbidden_indices,
                    extra={
                        "segment": segment,
                        "expected_tool": expected_tool,
                        "forbidden_tool": forbidden_tool,
                    },
                ),
            )

        if not expected_indices and not forbidden_indices:
            # Neither route appeared at all — segment is correct but the
            # routed family wasn't exercised. NOT_APPLICABLE rather than PASS.
            return self._result(invariant, trace, CheckOutcome.NOT_APPLICABLE)
        return self._result(invariant, trace, CheckOutcome.PASS)
