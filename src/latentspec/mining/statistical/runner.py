"""Track A driver — runs all four statistical sub-miners and returns candidates."""

from __future__ import annotations

from latentspec.schemas.invariant import InvariantCandidate
from latentspec.schemas.trace import NormalizedTrace

from latentspec.mining.statistical.anomaly import mine_anomaly_baselines
from latentspec.mining.statistical.association import mine_associations
from latentspec.mining.statistical.distribution import mine_distributions
from latentspec.mining.statistical.negative import mine_negatives
from latentspec.mining.statistical.sequence import mine_sequences


def run_statistical_track(
    traces: list[NormalizedTrace],
    *,
    min_support_sequence: float = 0.6,
) -> list[InvariantCandidate]:
    """Execute Track A end-to-end on a normalized trace batch.

    Returns the union of candidates from sequence (PrefixSpan with closed-pattern
    pruning), distribution, association, negative, and isolation-forest anomaly
    baseline miners. The orchestrator handles cross-validation against Track B
    and final confidence scoring.
    """
    candidates: list[InvariantCandidate] = []
    candidates.extend(mine_sequences(traces, min_support=min_support_sequence))
    candidates.extend(mine_distributions(traces))
    candidates.extend(mine_associations(traces))
    candidates.extend(mine_negatives(traces))
    candidates.extend(mine_anomaly_baselines(traces))
    return candidates
