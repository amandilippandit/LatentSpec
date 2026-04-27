"""Tests for TF-IDF embedding-based cross-validation merge."""

from __future__ import annotations

from latentspec.mining.confidence import cross_validate
from latentspec.mining.embeddings import (
    TfidfBackend,
    cluster_by_similarity,
    cluster_candidates_by_type_and_similarity,
)
from latentspec.models.invariant import InvariantType, Severity
from latentspec.schemas.invariant import InvariantCandidate


def _cand(
    type_: InvariantType,
    description: str,
    *,
    discovered_by: str,
    params: dict | None = None,
    support: float = 0.9,
    consistency: float = 0.9,
) -> InvariantCandidate:
    return InvariantCandidate(
        type=type_,
        description=description,
        formal_rule="...",
        support=support,
        consistency=consistency,
        severity=Severity.HIGH,
        discovered_by=discovered_by,  # type: ignore[arg-type]
        extra=params or {},
    )


def test_tfidf_clusters_paraphrases() -> None:
    docs = [
        "the agent always calls auth before db_write",
        "agent invokes auth prior to every db_write call",
        "tool latency stays under 500ms for 99% of latency",
        "tool latency p99 stays under 500ms latency",
    ]
    clusters = cluster_by_similarity(docs, threshold=0.2)
    # Two clusters: ordering paraphrases + latency paraphrases
    assert len(clusters) == 2
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [2, 2]


def test_tfidf_keeps_unrelated_apart() -> None:
    docs = [
        "agent calls auth before db_write",
        "user says refund -> escalate to human",
    ]
    clusters = cluster_by_similarity(docs, threshold=0.5)
    assert len(clusters) == 2


def test_cluster_respects_invariant_type() -> None:
    cands = [
        _cand(
            InvariantType.ORDERING,
            "the agent always invokes auth before db_write",
            discovered_by="statistical",
            params={"tool_a": "auth", "tool_b": "db_write"},
        ),
        _cand(
            InvariantType.STATISTICAL,
            "the agent always invokes auth before db_write",
            discovered_by="llm",
            params={"metric": "success_rate", "tool": "auth", "rate": 0.99},
        ),
    ]
    clusters = cluster_candidates_by_type_and_similarity(cands, threshold=0.2)
    # Different types must not merge even when descriptions identical.
    assert all(len(c) == 1 for c in clusters)


def test_cross_validate_merges_paraphrases_into_both() -> None:
    statistical = [
        _cand(
            InvariantType.ORDERING,
            "agent always calls auth before db_write",
            discovered_by="statistical",
            params={"tool_a": "auth", "tool_b": "db_write"},
        )
    ]
    llm = [
        _cand(
            InvariantType.ORDERING,
            "the agent invokes auth prior to every db_write call",
            discovered_by="llm",
            params={"tool_a": "auth", "tool_b": "db_write"},
        )
    ]
    merged = cross_validate(statistical, llm, similarity_threshold=0.2)
    assert len(merged) == 1
    assert merged[0].discovered_by == "both"


def test_cross_validate_does_not_merge_when_params_conflict() -> None:
    statistical = [
        _cand(
            InvariantType.ORDERING,
            "agent always calls auth before db_write",
            discovered_by="statistical",
            params={"tool_a": "auth", "tool_b": "db_write"},
        )
    ]
    llm = [
        _cand(
            InvariantType.ORDERING,
            "agent always calls auth before db_write",
            discovered_by="llm",
            # different tool_b — prose collides but params disagree
            params={"tool_a": "auth", "tool_b": "audit_log"},
        )
    ]
    merged = cross_validate(statistical, llm, similarity_threshold=0.2)
    assert len(merged) == 2
