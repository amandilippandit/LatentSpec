"""§3.4 confidence scoring + cross-validation + three-band gating.

The formula from the technical plan:

    Confidence = 0.4 * support_score
               + 0.3 * consistency_score
               + 0.2 * cross_val_bonus
               + 0.1 * clarity_score

Three-band gating:
    < 0.6  -> rejected (auto)
    0.6-0.8 -> pending (human review)
    > 0.8  -> active (auto)

Cross-validation rule: candidates found by *both* tracks (statistical AND
LLM) earn the cross_val_bonus. We decide "same rule" via TF-IDF cosine
clustering over descriptions (`mining/embeddings.py`) — two LLM
rewordings of the same logical rule cluster together rather than being
emitted twice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from latentspec.mining.embeddings import (
    EmbeddingBackend,
    cluster_candidates_by_type_and_similarity,
)
from latentspec.models.invariant import InvariantStatus
from latentspec.schemas.invariant import InvariantCandidate


@dataclass(frozen=True)
class ConfidenceWeights:
    """Per §3.4 weighted-sum formula."""

    support: float = 0.4
    consistency: float = 0.3
    cross_val: float = 0.2
    clarity: float = 0.1

    def __post_init__(self) -> None:
        total = self.support + self.consistency + self.cross_val + self.clarity
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"weights must sum to 1.0, got {total}")


DEFAULT_WEIGHTS = ConfidenceWeights()


# ----- Score components ---------------------------------------------------


def _clarity_score(description: str) -> float:
    """Heuristic semantic-clarity score for an invariant description.

    The ideal description is:
      - short enough to read at a glance (≤ 140 chars)
      - specific (mentions concrete tool/keyword/threshold names)
      - free of formal-methods jargon (per §6.2)
    """
    if not description:
        return 0.0
    desc = description.strip()
    score = 1.0

    # Length: penalise very short or very long descriptions
    n = len(desc)
    if n < 25:
        score -= 0.4
    elif n > 200:
        score -= 0.2

    # Specificity: presence of a backticked identifier or a number
    has_identifier = bool(re.search(r"`[^`]+`", desc))
    has_number = bool(re.search(r"\d", desc))
    if not (has_identifier or has_number):
        score -= 0.2

    # Jargon penalty: formal-methods vocabulary leaking into UI text
    jargon_terms = (
        "predicate",
        "smt",
        "z3",
        "temporal logic",
        "first-order",
        "forall",
        "exists(",
        "isolation forest",
        "prefixspan",
    )
    if any(term in desc.lower() for term in jargon_terms):
        score -= 0.3

    return max(0.0, min(1.0, score))


def _params_overlap(a: dict, b: dict) -> bool:
    """Two candidates also need params overlap to merge.

    Cosine similarity over descriptions catches rewordings, but we still
    require the structured rule data to be compatible — otherwise an LLM
    description with different `tool_a` / `tool_b` could merge with a
    statistical candidate just because the prose is similar.
    """
    if not a or not b:
        return True
    common_keys = (
        "tool_a", "tool_b", "tool", "keyword", "metric",
        "terminator_tool", "upstream_tool", "downstream_tool",
        "expected_tool", "segment", "feature",
    )
    for k in common_keys:
        if k in a and k in b and a[k] != b[k]:
            return False
    return True


def _merge_cluster(cluster: list[InvariantCandidate]) -> InvariantCandidate:
    """Combine N candidates that the embedding clusterer flagged as
    semantically equivalent. Take the max support / consistency, union
    evidence, prefer the longest description."""
    cluster = list(cluster)
    if len(cluster) == 1:
        return cluster[0]

    base = max(cluster, key=lambda c: len(c.description))
    sources = {c.discovered_by for c in cluster}
    discovered_by = "both" if {"statistical", "llm"} <= sources else next(iter(sources))

    merged_evidence: list[str] = []
    seen: set[str] = set()
    for c in cluster:
        for tid in c.evidence_trace_ids:
            if tid in seen:
                continue
            merged_evidence.append(tid)
            seen.add(tid)
            if len(merged_evidence) >= 100:
                break

    # Use the most-specific (most populated) params blob as the merged blob.
    merged_extra: dict = {}
    for c in cluster:
        merged_extra.update(c.extra)

    return base.model_copy(
        update={
            "support": max(c.support for c in cluster),
            "consistency": max(c.consistency for c in cluster),
            "evidence_trace_ids": merged_evidence,
            "discovered_by": discovered_by,
            "extra": merged_extra,
        }
    )


def cross_validate(
    statistical: list[InvariantCandidate],
    llm: list[InvariantCandidate],
    *,
    similarity_threshold: float = 0.7,
    backend: EmbeddingBackend | None = None,
) -> list[InvariantCandidate]:
    """Merge two candidate lists via TF-IDF cosine clustering.

    Two candidates merge into one when:
      1. They share the same `type`, AND
      2. Their description vectors have cosine similarity >= threshold, AND
      3. Their structured `params` are compatible (no conflicting keys).

    Candidates that end up in a cluster spanning both tracks are marked
    `discovered_by="both"` and earn the §3.4 cross_val_bonus.
    """
    all_candidates: list[InvariantCandidate] = list(statistical) + list(llm)
    if not all_candidates:
        return []

    clusters = cluster_candidates_by_type_and_similarity(
        all_candidates, threshold=similarity_threshold, backend=backend
    )

    merged: list[InvariantCandidate] = []
    for cluster_indices in clusters:
        members = [all_candidates[i] for i in cluster_indices]
        # If any pair of members has incompatible params, fall back to
        # the original normalize-string match (so we don't accidentally
        # collapse two different rules with similar wording).
        compatible = True
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if not _params_overlap(members[i].extra, members[j].extra):
                    compatible = False
                    break
            if not compatible:
                break

        if compatible:
            merged.append(_merge_cluster(members))
        else:
            # Emit each separately; they're not the same rule.
            merged.extend(members)

    return merged


# ----- Final confidence + triage -----------------------------------------


@dataclass(frozen=True)
class ScoreBreakdown:
    support: float
    consistency: float
    cross_val: float
    clarity: float
    final: float


def score_candidate(
    candidate: InvariantCandidate,
    *,
    weights: ConfidenceWeights = DEFAULT_WEIGHTS,
) -> ScoreBreakdown:
    """Apply the §3.4 weighted-sum formula to a single candidate."""
    support_score = max(0.0, min(1.0, candidate.support))
    consistency_score = max(0.0, min(1.0, candidate.consistency))
    cross_val_bonus = 1.0 if candidate.discovered_by == "both" else 0.0
    clarity = _clarity_score(candidate.description)

    final = (
        weights.support * support_score
        + weights.consistency * consistency_score
        + weights.cross_val * cross_val_bonus
        + weights.clarity * clarity
    )
    return ScoreBreakdown(
        support=support_score,
        consistency=consistency_score,
        cross_val=cross_val_bonus,
        clarity=clarity,
        final=round(final, 4),
    )


def triage(
    confidence: float,
    *,
    reject_threshold: float = 0.6,
    review_threshold: float = 0.8,
) -> InvariantStatus:
    """§3.4 three-band gating policy.

    Args:
        confidence: final score on [0, 1].
        reject_threshold: anything strictly below is auto-rejected.
        review_threshold: anything at or above is auto-activated;
            scores in [reject, review) become PENDING for human review.
    """
    if confidence < reject_threshold:
        return InvariantStatus.REJECTED
    if confidence < review_threshold:
        return InvariantStatus.PENDING
    return InvariantStatus.ACTIVE
