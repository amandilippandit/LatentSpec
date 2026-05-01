"""Stage 2 invariant mining (§3.2).

Two parallel tracks that cross-validate each other:
  - Track A (statistical): PrefixSpan sequences + distributional analysis
    + association rules + isolation-forest anomaly baselines.
  - Track B (LLM): semantic analysis via Claude with a structured
    extraction prompt over batches of 50–100 normalized traces.

The orchestrator merges both tracks via §3.4 confidence scoring with the
three-band gating policy (>0.8 active / 0.6–0.8 pending / <0.6 rejected).
"""

from latentspec.mining.confidence import (
    ConfidenceWeights,
    cross_validate,
    score_candidate,
    triage,
)
from latentspec.mining.formalization import formalize, generate_formal_rule
from latentspec.mining.orchestrator import MiningResult, mine_invariants

__all__ = [
    "ConfidenceWeights",
    "MiningResult",
    "cross_validate",
    "formalize",
    "generate_formal_rule",
    "mine_invariants",
    "score_candidate",
    "triage",
]
