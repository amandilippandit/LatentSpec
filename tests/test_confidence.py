"""Tests for §3.4 confidence scoring + cross-validation + three-band gating."""

from __future__ import annotations

import pytest

from latentspec.mining.confidence import (
    ConfidenceWeights,
    cross_validate,
    score_candidate,
    triage,
)
from latentspec.models.invariant import InvariantStatus, InvariantType, Severity
from latentspec.schemas.invariant import InvariantCandidate


def _cand(
    *,
    description: str,
    support: float,
    consistency: float,
    discovered_by: str,
    type_: InvariantType = InvariantType.ORDERING,
) -> InvariantCandidate:
    return InvariantCandidate(
        type=type_,
        description=description,
        formal_rule=f"forall trace: holds('{description[:30]}')",
        support=support,
        consistency=consistency,
        severity=Severity.HIGH,
        discovered_by=discovered_by,  # type: ignore[arg-type]
    )


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        ConfidenceWeights(support=0.5, consistency=0.3, cross_val=0.2, clarity=0.5)


def test_score_breakdown_matches_paper_formula() -> None:
    cand = _cand(
        description="Always calls `check_inventory` before `create_order`",
        support=0.95,
        consistency=0.97,
        discovered_by="both",
    )
    bd = score_candidate(cand)
    # 0.4*0.95 + 0.3*0.97 + 0.2*1.0 + 0.1*clarity
    expected_baseline = 0.4 * 0.95 + 0.3 * 0.97 + 0.2 * 1.0
    assert bd.final >= expected_baseline
    assert bd.cross_val == 1.0


def test_cross_val_bonus_only_for_both_tracks() -> None:
    cand = _cand(
        description="Always calls `auth` before `db_write`",
        support=0.9,
        consistency=0.9,
        discovered_by="statistical",
    )
    bd = score_candidate(cand)
    assert bd.cross_val == 0.0


def test_cross_validate_marks_overlap() -> None:
    a = _cand(description="Always calls `auth` before `db_write`",
              support=0.9, consistency=0.9, discovered_by="statistical")
    b = _cand(description="always calls `auth` before `db_write`",
              support=0.85, consistency=0.95, discovered_by="llm")
    merged = cross_validate([a], [b])
    assert len(merged) == 1
    assert merged[0].discovered_by == "both"
    # Max-take semantics: support ≥ original support
    assert merged[0].support >= 0.9
    assert merged[0].consistency >= 0.95


def test_triage_three_bands() -> None:
    assert triage(0.55) == InvariantStatus.REJECTED
    assert triage(0.7) == InvariantStatus.PENDING
    assert triage(0.9) == InvariantStatus.ACTIVE
    # Boundary values
    assert triage(0.6) == InvariantStatus.PENDING
    assert triage(0.8) == InvariantStatus.ACTIVE


def test_clarity_penalises_jargon() -> None:
    plain = _cand(
        description="Agent always calls `check_inventory` before `create_order`",
        support=0.9, consistency=0.9, discovered_by="both",
    )
    jargon = _cand(
        description="forall trace: predicate(temporal logic) holds(SMT, Z3)",
        support=0.9, consistency=0.9, discovered_by="both",
    )
    assert score_candidate(plain).clarity > score_candidate(jargon).clarity
