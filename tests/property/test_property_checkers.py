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

