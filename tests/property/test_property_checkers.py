"""Property-based tests for the checker dispatch.

The invariants we assert (which must hold for ANY generated input):

  - dispatch never raises an unhandled exception. Worst case is the
    runner converts CheckerError to NOT_APPLICABLE.
  - duration_ms is always non-negative and below the streaming budget
    (100ms) for rule-based checkers on bounded traces.
  - outcome ∈ valid CheckOutcome enum members.
  - dispatch is deterministic — calling it twice on the same input
    produces the same outcome.
"""

from __future__ import annotations

import time

from hypothesis import given, settings, HealthCheck, strategies as st
from hypothesis import seed

from latentspec.checking import dispatch
from latentspec.checking.base import CheckOutcome
from latentspec.checking.runner import check_trace
from latentspec.models.invariant import InvariantType
from tests.property.strategies import invariant_spec, normalized_trace


VALID_OUTCOMES = set(CheckOutcome)


@given(spec=invariant_spec(), trace=normalized_trace())
@settings(max_examples=200, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_dispatch_never_raises(spec, trace) -> None:
    """For any generated (spec, trace), dispatch produces a valid result.

    `dispatch` itself can raise CheckerError on malformed params — but the
    runner-level `check_trace` MUST always return a list of valid results.
    """
    results = check_trace([spec], trace)
    assert len(results) == 1
    r = results[0]
    assert r.outcome in VALID_OUTCOMES
    assert r.duration_ms >= 0
    # Rule-based checkers have a 100ms budget; LLM judge runs offline so
    # we exempt OUTPUT_FORMAT from the latency assertion.
    if r.invariant_type != InvariantType.OUTPUT_FORMAT:
        assert r.duration_ms < 100, f"checker exceeded 100ms budget: {r.duration_ms}"


@given(spec=invariant_spec(), trace=normalized_trace())
@settings(max_examples=100, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_dispatch_is_deterministic(spec, trace) -> None:
    a = check_trace([spec], trace)[0].outcome
    b = check_trace([spec], trace)[0].outcome
    assert a == b


@given(spec_a=invariant_spec(type_=InvariantType.ORDERING),
       spec_b=invariant_spec(type_=InvariantType.ORDERING),
       trace=normalized_trace())
@settings(max_examples=100, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_independent_invariants_dont_cross_contaminate(spec_a, spec_b, trace) -> None:
    """Running two independent specs against the same trace must produce
    the same result as running each one separately."""
    joint = check_trace([spec_a, spec_b], trace)
    a_alone = check_trace([spec_a], trace)
    b_alone = check_trace([spec_b], trace)
    assert joint[0].outcome == a_alone[0].outcome
    assert joint[1].outcome == b_alone[0].outcome


# ---- specific per-type invariants ---------------------------------------


@given(trace=normalized_trace())
@settings(max_examples=200, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_negative_repertoire_subset_always_passes(trace) -> None:
    """If `allowed_repertoire` is a superset of every tool the trace uses,
    the negative checker MUST return PASS or NOT_APPLICABLE — never FAIL."""
    from latentspec.checking.base import InvariantSpec
    from latentspec.models.invariant import Severity
    from latentspec.schemas.trace import ToolCallStep

    used = {s.tool for s in trace.steps if isinstance(s, ToolCallStep)}
    if not used:
        return  # no tool calls ⇒ trivially holds
    spec = InvariantSpec(
        id="inv-neg-superset",
        type=InvariantType.NEGATIVE,
        description="superset",
        formal_rule="...",
        severity=Severity.MEDIUM,
        params={"allowed_repertoire": sorted(used) + ["__sentinel__"]},
    )
    outcome = dispatch(spec, trace).outcome
    assert outcome in {CheckOutcome.PASS, CheckOutcome.NOT_APPLICABLE}, (
        f"superset repertoire failed: outcome={outcome}, used={used}"
    )


@given(trace=normalized_trace(min_steps=2, max_steps=20))
@settings(max_examples=200, deadline=2000, suppress_health_check=[HealthCheck.too_slow])
def test_ordering_consistent_with_tool_order(trace) -> None:
    """If we look at the literal order of tool calls in the trace and
    pick a pair (A, B) where A appears strictly before B somewhere,
    ordering(A, B) must PASS."""
    from latentspec.checking.base import InvariantSpec
    from latentspec.models.invariant import Severity
    from latentspec.schemas.trace import ToolCallStep

    tool_indices: list[tuple[int, str]] = [
        (i, s.tool) for i, s in enumerate(trace.steps) if isinstance(s, ToolCallStep)
    ]
    if len(tool_indices) < 2:
        return
    # find the first pair where index_a < index_b and tools differ
    a_pair = None
    for i, (ai, atool) in enumerate(tool_indices):
        for bj, btool in tool_indices[i + 1 :]:
            if atool != btool:
                a_pair = (atool, btool)
                break
        if a_pair:
            break
    if a_pair is None:
        return

    spec = InvariantSpec(
        id="inv-ord-witness",
        type=InvariantType.ORDERING,
        description="witnessed",
        formal_rule="...",
        severity=Severity.HIGH,
        params={"tool_a": a_pair[0], "tool_b": a_pair[1]},
    )
    outcome = dispatch(spec, trace).outcome
    # The first occurrence of a_pair[0] is before SOME occurrence of a_pair[1]
    # -> can be PASS (first a precedes first b) or NOT_APPLICABLE (b absent
    # from this slice's "first a"). FAIL is impossible.
    assert outcome != CheckOutcome.FAIL, f"ordering ({a_pair[0]} -> {a_pair[1]}) failed"
