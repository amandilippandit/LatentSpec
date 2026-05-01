"""Driver — checks one trace (or a set) against a list of invariants."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable

from latentspec.checking.base import (
    CheckOutcome,
    CheckResult,
    CheckerError,
    InvariantSpec,
)
from latentspec.checking.dispatch import dispatch
from latentspec.models.invariant import Invariant
from latentspec.schemas.trace import NormalizedTrace

log = logging.getLogger(__name__)


def invariant_to_spec(inv: Invariant) -> InvariantSpec:
    """Convert a DB row to the lightweight checker contract.

    Params are validated against the per-type Pydantic schema before the
    checker sees them. A row whose params don't validate ends up with an
    empty `params` dict — which the checker recognises and returns
    NOT_APPLICABLE for, rather than blowing up at runtime.
    """
    from latentspec.schemas.params import ParamsValidationError, validate_params

    try:
        params = validate_params(inv.type, dict(inv.params or {}))
    except ParamsValidationError as e:
        log.warning("invariant %s has invalid params: %s", inv.id, e)
        params = {}
    return InvariantSpec(
        id=str(inv.id) if isinstance(inv.id, uuid.UUID) else str(inv.id),
        type=inv.type,
        description=inv.description,
        formal_rule=inv.formal_rule,
        severity=inv.severity,
        params=params,
    )


def check_trace(
    invariants: Iterable[InvariantSpec],
    trace: NormalizedTrace,
) -> list[CheckResult]:
    """Run every active invariant against one trace."""
    results: list[CheckResult] = []
    for inv in invariants:
        try:
            results.append(dispatch(inv, trace))
        except CheckerError as e:
            log.warning("checker error for %s: %s", inv.id, e)
            results.append(
                CheckResult(
                    invariant_id=inv.id,
                    invariant_type=inv.type,
                    invariant_description=inv.description,
                    severity=inv.severity,
                    trace_id=trace.trace_id,
                    outcome=CheckOutcome.NOT_APPLICABLE,
                )
            )
    return results


def check_traces(
    invariants: Iterable[InvariantSpec],
    traces: Iterable[NormalizedTrace],
) -> list[CheckResult]:
    """Run every invariant against every trace; flat result list."""
    invariants = list(invariants)
    out: list[CheckResult] = []
    for trace in traces:
        out.extend(check_trace(invariants, trace))
    return out
