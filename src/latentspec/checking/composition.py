"""Composition invariant checker (§3.3).

`upstream_tool` must precede every `downstream_tool` invocation. Maps to
the doc's "Multi-agent: Agent B waits for Agent A's output" example when
multi-agent traces are interleaved into a single normalized trace.

A more elaborate variant (cross-trace composition over multi-agent runs)
ships in the Expansion phase along with explicit multi-agent invariants.
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


class CompositionChecker(Checker):
    invariant_type = InvariantType.COMPOSITION

    def check(
        self, invariant: InvariantSpec, trace: NormalizedTrace
    ) -> CheckResult:
        upstream = invariant.params.get("upstream_tool")
        downstream = invariant.params.get("downstream_tool")
        if not upstream or not downstream:
            raise CheckerError(
                f"composition invariant {invariant.id} needs upstream/downstream tools"
            )

        upstream_seen = False
        downstream_indices: list[int] = []

        for idx, step in enumerate(trace.steps):
            if not isinstance(step, ToolCallStep):
                continue
            if step.tool == upstream:
                upstream_seen = True
            elif step.tool == downstream:
                if not upstream_seen:
                    return self._result(
                        invariant,
                        trace,
                        CheckOutcome.FAIL,
                        ViolationDetails(
                            expected=f"`{upstream}` before any `{downstream}`",
                            observed=(
                                f"`{downstream}` at step {idx} ran with no preceding `{upstream}`"
                            ),
                            affected_step_indices=[idx],
                            extra={
                                "upstream_tool": upstream,
                                "downstream_tool": downstream,
                            },
                        ),
                    )
                downstream_indices.append(idx)

        if not downstream_indices:
            return self._result(invariant, trace, CheckOutcome.NOT_APPLICABLE)
        return self._result(invariant, trace, CheckOutcome.PASS)
