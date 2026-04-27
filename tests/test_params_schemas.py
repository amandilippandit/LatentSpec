"""Tests for the per-type params Pydantic schemas (§3.3 schema enforcement)."""

from __future__ import annotations

import pytest

from latentspec.models.invariant import InvariantType
from latentspec.schemas.params import (
    OrderingParams,
    ParamsValidationError,
    StatisticalParams,
    validate_params,
)


def test_ordering_accepts_well_formed() -> None:
    out = validate_params(InvariantType.ORDERING, {"tool_a": "auth", "tool_b": "db_write"})
    assert out["tool_a"] == "auth"
    assert out["tool_b"] == "db_write"


def test_ordering_rejects_missing_field() -> None:
    with pytest.raises(ParamsValidationError):
        validate_params(InvariantType.ORDERING, {"tool_a": "auth"})


def test_ordering_rejects_invalid_tool_name() -> None:
    with pytest.raises(ParamsValidationError):
        validate_params(InvariantType.ORDERING, {"tool_a": "with spaces", "tool_b": "ok"})


def test_statistical_latency_requires_threshold() -> None:
    with pytest.raises(ParamsValidationError):
        validate_params(
            InvariantType.STATISTICAL,
            {"metric": "latency_ms", "tool": "search"},
        )


def test_statistical_success_requires_rate() -> None:
    with pytest.raises(ParamsValidationError):
        validate_params(
            InvariantType.STATISTICAL,
            {"metric": "success_rate", "tool": "book"},
        )


def test_statistical_envelope_validates() -> None:
    out = validate_params(
        InvariantType.STATISTICAL,
        {"metric": "feature_envelope", "feature": "step_count", "p1": 1.0, "p99": 50.0},
    )
    assert out["metric"] == "feature_envelope"


def test_statistical_envelope_rejects_invalid_range() -> None:
    with pytest.raises(ParamsValidationError):
        validate_params(
            InvariantType.STATISTICAL,
            {"metric": "feature_envelope", "feature": "step_count", "p1": 50.0, "p99": 1.0},
        )


def test_negative_requires_exactly_one_mode() -> None:
    # Both modes set: invalid
    with pytest.raises(ParamsValidationError):
        validate_params(
            InvariantType.NEGATIVE,
            {
                "forbidden_patterns": ["delete"],
                "allowed_repertoire": ["search_flights"],
            },
        )
    # Neither mode set: invalid
    with pytest.raises(ParamsValidationError):
        validate_params(InvariantType.NEGATIVE, {})


def test_negative_closed_world_auto_flag() -> None:
    out = validate_params(
        InvariantType.NEGATIVE,
        {"allowed_repertoire": ["search_flights", "book_flight"]},
    )
    assert out["closed_world"] is True


def test_state_requires_non_empty_forbidden_after() -> None:
    with pytest.raises(ParamsValidationError):
        validate_params(
            InvariantType.STATE,
            {"terminator_tool": "session_close", "forbidden_after": []},
        )


def test_tool_selection_rejects_invalid_segment() -> None:
    with pytest.raises(ParamsValidationError):
        validate_params(
            InvariantType.TOOL_SELECTION,
            {
                "segment": "EU customer",  # space invalid
                "expected_tool": "payments_v2",
            },
        )


def test_validation_drops_bad_candidate_in_formalization() -> None:
    """A candidate with malformed extra is dropped by formalize()."""
    from datetime import UTC, datetime

    from latentspec.mining.formalization import formalize
    from latentspec.models.invariant import Severity
    from latentspec.schemas.invariant import InvariantCandidate

    bad = InvariantCandidate(
        type=InvariantType.ORDERING,
        description="placeholder",
        formal_rule="forall trace: ...",
        support=0.95,
        consistency=0.95,
        severity=Severity.HIGH,
        discovered_by="statistical",
        extra={"tool_a": "with spaces", "tool_b": "ok"},
    )
    assert formalize(bad) is None
