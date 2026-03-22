"""Type-keyed checker dispatcher.

Maps each InvariantType to a Checker instance. New types plug in via
`register()`. Today seven of the eight §3.3 types have rule-based checkers;
output_format uses the LLM judge.
"""

from __future__ import annotations

import time

from latentspec.checking.base import (
    Checker,
    CheckOutcome,
    CheckResult,
    InvariantSpec,
)
from latentspec.checking.composition import CompositionChecker
from latentspec.checking.conditional import ConditionalChecker
from latentspec.checking.llm_judge import LLMJudgeChecker
from latentspec.checking.negative import NegativeChecker
from latentspec.checking.ordering import OrderingChecker
from latentspec.checking.state import StateChecker
from latentspec.checking.statistical import StatisticalChecker
from latentspec.checking.tool_selection import ToolSelectionChecker
from latentspec.models.invariant import InvariantType
from latentspec.schemas.trace import NormalizedTrace


_REGISTRY: dict[InvariantType, Checker] = {}


def register(checker: Checker) -> None:
    _REGISTRY[checker.invariant_type] = checker


def get_checker(invariant_type: InvariantType) -> Checker | None:
    return _REGISTRY.get(invariant_type)


def dispatch(
    invariant: InvariantSpec, trace: NormalizedTrace
) -> CheckResult:
    """Route to the registered checker; record duration_ms."""
    checker = get_checker(invariant.type)
    if checker is None:
        return CheckResult(
            invariant_id=invariant.id,
            invariant_type=invariant.type,
            invariant_description=invariant.description,
            severity=invariant.severity,
            trace_id=trace.trace_id,
            outcome=CheckOutcome.NOT_APPLICABLE,
        )

    start = time.perf_counter()
    result = checker.check(invariant, trace)
    result.duration_ms = round((time.perf_counter() - start) * 1000, 3)
    return result


# Default registry: 7 rule-based + 1 LLM judge — full §3.3 coverage.
register(OrderingChecker())
register(ConditionalChecker())
register(NegativeChecker())
register(StatisticalChecker())
register(StateChecker())
register(CompositionChecker())
register(ToolSelectionChecker())
register(LLMJudgeChecker())
