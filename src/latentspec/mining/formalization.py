"""Stage 3 — invariant formalization (§3.2).

Converts a scored InvariantCandidate into a `MinedInvariant` (the §3.2 JSON
shape) ready for DB persistence. Per §10.1, the `formal_rule` is stored as
a Python predicate string today; the Z3 SMT compilation graduates to a
visible feature in month 6+ when enterprise demand pulls it forward.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from latentspec.mining.confidence import (
    DEFAULT_WEIGHTS,
    ConfidenceWeights,
    score_candidate,
    triage,
)
from latentspec.models.invariant import InvariantStatus
from latentspec.schemas.invariant import InvariantCandidate, MinedInvariant
from latentspec.schemas.params import ParamsValidationError, validate_params


def generate_formal_rule(candidate: InvariantCandidate) -> str:
    """Return the formalized predicate string.

    Today this is whatever the discovering miner produced. The Z3 backend
    consumes this same string downstream (see §10.1 for the staged unveiling
    of formal verification as a premium feature). Centralizing the function
    here makes the future SMT-compilation upgrade a single-file change.
    """
    rule = candidate.formal_rule.strip()
    if rule:
        return rule
    # Fallback: derive a placeholder from the description so we never
    # persist an empty formal_rule (which would break the checker).
    return f"holds(trace, '{candidate.description.strip()[:120]}')"


def formalize(
    candidate: InvariantCandidate,
    *,
    reject_threshold: float = 0.6,
    review_threshold: float = 0.8,
    weights: ConfidenceWeights = DEFAULT_WEIGHTS,
) -> MinedInvariant | None:
    """Stage 3 — validate params against the type schema, then score and triage.

    Returns None when params validation fails (the candidate is dropped from
    the active set rather than persisted with malformed shape that the
    runtime checker can't read).
    """
    try:
        params = validate_params(candidate.type, dict(candidate.extra))
    except ParamsValidationError:
        # Drop the candidate — better to lose a rule than persist one the
        # runtime checker can't evaluate. Track-A bugs surface as test
        # failures; Track-B (LLM) drift surfaces as a count delta.
        return None

    breakdown = score_candidate(candidate, weights=weights)
    status: InvariantStatus = triage(
        breakdown.final,
        reject_threshold=reject_threshold,
        review_threshold=review_threshold,
    )

    return MinedInvariant(
        invariant_id=f"inv-{uuid.uuid4().hex[:8]}",
        type=candidate.type,
        description=candidate.description.strip(),
        formal_rule=generate_formal_rule(candidate),
        confidence=breakdown.final,
        support_score=breakdown.support,
        consistency_score=breakdown.consistency,
        cross_val_bonus=breakdown.cross_val * weights.cross_val,
        clarity_score=breakdown.clarity,
        evidence_count=len(candidate.evidence_trace_ids),
        violation_count=0,
        discovered_at=datetime.now(UTC),
        status=status,
        severity=candidate.severity,
        discovered_by=candidate.discovered_by,  # type: ignore[arg-type]
        evidence_trace_ids=list(dict.fromkeys(candidate.evidence_trace_ids))[:100],
        params=params,
    )
