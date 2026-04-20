"""Reference (oracle) implementations of each checker — independently coded.

Differential testing strategy: each function here implements the checker
semantics from a different angle than the production code (e.g. counters
+ list comprehensions vs the production's index-tracking + early-return).
Cross-checking the two implementations against the same generated input
exposes bugs in either one.

These are deliberately simpler / slower than the production checkers — we
trade speed for "obviously correct" code so disagreements point at the
optimised path rather than the reference.
"""

from __future__ import annotations

from latentspec.checking.base import (
    CheckOutcome,
    InvariantSpec,
)
from latentspec.schemas.trace import (
    NormalizedTrace,
    ToolCallStep,
    UserInputStep,
)


def ref_ordering(spec: InvariantSpec, trace: NormalizedTrace) -> CheckOutcome:
    """Reference: explicitly enumerate (i, j) pairs."""
    a = spec.params.get("tool_a")
    b = spec.params.get("tool_b")
    if not a or not b:
        return CheckOutcome.NOT_APPLICABLE

    tool_steps = [
        (i, s.tool) for i, s in enumerate(trace.steps) if isinstance(s, ToolCallStep)
    ]
    has_b = any(t == b for _, t in tool_steps)
    if not has_b:
        return CheckOutcome.NOT_APPLICABLE

    for j, tool in tool_steps:
        if tool != b:
            continue
        # Earliest occurrence of `b` — every occurrence of `b` must have an
        # `a` somewhere before it. The simplest semantics is: there must
        # exist some `a` before the FIRST `b`. Both production and reference
        # use this contract.
        for i, prior in tool_steps:
            if i >= j:
                break
            if prior == a:
                return CheckOutcome.PASS
        return CheckOutcome.FAIL
    return CheckOutcome.NOT_APPLICABLE


def ref_conditional(spec: InvariantSpec, trace: NormalizedTrace) -> CheckOutcome:
    """Reference: use a set of all user-input tokens then a simple membership."""
    keyword = (spec.params.get("keyword") or "").lower()
    tool = spec.params.get("tool")
    if not keyword or not tool:
        return CheckOutcome.NOT_APPLICABLE

    user_text = " ".join(
        s.content for s in trace.steps if isinstance(s, UserInputStep)
    ).lower()
    if keyword not in user_text:
        return CheckOutcome.NOT_APPLICABLE

    for s in trace.steps:
        if isinstance(s, ToolCallStep) and s.tool == tool:
            return CheckOutcome.PASS
    return CheckOutcome.FAIL


def ref_negative(spec: InvariantSpec, trace: NormalizedTrace) -> CheckOutcome:
    patterns = spec.params.get("forbidden_patterns") or []
    allowed = spec.params.get("allowed_repertoire") or []

    tools_called = [
        s.tool for s in trace.steps if isinstance(s, ToolCallStep)
    ]

    if allowed:
        allowed_set = set(allowed)
        for tool in tools_called:
            if tool not in allowed_set:
                return CheckOutcome.FAIL
        return CheckOutcome.PASS

    if patterns:
        patterns_lower = [str(p).lower() for p in patterns]
        for tool in tools_called:
            tool_lc = tool.lower()
            for pat in patterns_lower:
