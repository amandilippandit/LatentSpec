"""Track A: statistical pattern mining (§3.2).

Components:
  - sequence: PrefixSpan-style frequent subsequence mining over tool calls
              → ordering invariants
  - distribution: latency / response-length distributional analysis
              → statistical invariants (p99 thresholds)
  - association: input-feature → tool-choice association rules
              → conditional invariants
  - negative: absence detection (frequent forbidden actions)
              → negative invariants
"""

from latentspec.mining.statistical.anomaly import mine_anomaly_baselines
from latentspec.mining.statistical.association import mine_associations
from latentspec.mining.statistical.distribution import mine_distributions
from latentspec.mining.statistical.negative import mine_negatives
from latentspec.mining.statistical.sequence import mine_sequences
from latentspec.mining.statistical.runner import run_statistical_track

__all__ = [
    "mine_anomaly_baselines",
    "mine_associations",
    "mine_distributions",
    "mine_negatives",
    "mine_sequences",
    "run_statistical_track",
]
