"""Per-agent threshold calibrator.

Inputs: a representative trace sample for the agent.
Outputs: `CalibratedThresholds` ready to overwrite the defaults.

The calibration is data-driven, not magic-constant-driven:

  - **min_support** uses an elbow detection on the cumulative pattern
    frequency curve. We mine all patterns at min_support=0.05 (very
    permissive), sort by frequency descending, and pick the support
    value where adding 10 more patterns adds no incremental information
    (Kneedle algorithm, simplified inline).

  - **min_mi_bits** uses Benjamini-Hochberg FDR control. We mine all
    keyword/tool pairs at min_mi=0, compute their chi-square p-values,
    apply BH at `target_fdr`, and the effective MI threshold is the
    smallest MI that survives.

  - **review threshold** is bisected so the expected queue size
    (calibrated_invariants × P(score in [reject, review))) lands in
    [target_queue_min, target_queue_max].

  - **chi-square threshold** uses the actual fingerprint cardinality:
    look up the χ² critical value for df = (k-1) where k is the
    number of distinct fingerprints observed.

  - **ph_threshold** is calibrated by simulating the detector on the
    training-set pass-rate sequence and picking the smallest threshold
    that doesn't fire on the (assumed-clean) training data.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from latentspec.mining.fingerprint import fingerprint
from latentspec.mining.statistical.runner import run_statistical_track
from latentspec.schemas.trace import NormalizedTrace, ToolCallStep, UserInputStep


# ---- result types -------------------------------------------------------


@dataclass
class DistributionSummary:
    """Captured per-agent statistics the calibrator persists alongside results."""

    n_traces: int
    n_distinct_tools: int
    n_distinct_fingerprints: int
    median_steps_per_trace: float
    p99_steps_per_trace: float
    median_tool_calls_per_trace: float
    pattern_support_curve: list[float] = field(default_factory=list)


@dataclass
class CalibratedThresholds:
    mining_min_support: float
    mining_min_directionality: float
    mining_max_path_length: int
    association_min_mi_bits: float
    association_min_lift: float
    association_min_keyword_traces: int
    statistical_p_target: float
    anomaly_contamination: float
    confidence_reject_threshold: float
    confidence_review_threshold: float
    fingerprint_chi_square_threshold: float
    drift_ph_threshold: float
    drift_cusum_threshold: float
    distribution_summary: DistributionSummary


# ---- chi-square critical values -----------------------------------------


# χ² critical values @ p<0.01 for df=1..30, then we extrapolate
_CHI2_001 = {
    1: 6.63, 2: 9.21, 3: 11.34, 4: 13.28, 5: 15.09,
    6: 16.81, 7: 18.48, 8: 20.09, 9: 21.67, 10: 23.21,
    15: 30.58, 20: 37.57, 25: 44.31, 30: 50.89,
}


def _chi2_critical(df: int, *, alpha: float = 0.01) -> float:
    """Critical χ² value at p<alpha. Uses tabulated values for small df,
    Wilson-Hilferty approximation for larger df."""
    if df <= 0:
        return _CHI2_001[1]
    if df in _CHI2_001:
        return _CHI2_001[df]
    # Linear interpolation between tabulated values
    keys = sorted(_CHI2_001)
    if df < keys[-1]:
        for k1, k2 in zip(keys, keys[1:], strict=True):
            if k1 <= df <= k2:
                t = (df - k1) / (k2 - k1)
                return _CHI2_001[k1] + t * (_CHI2_001[k2] - _CHI2_001[k1])
    # Wilson-Hilferty approximation:  χ²(df) ≈ df · (1 - 2/(9df) + z·√(2/(9df)))³
    # z=2.326 corresponds to p<0.01
    z = 2.326
    factor = 1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))
    return df * (factor ** 3)


# ---- elbow detection (simplified Kneedle) -------------------------------


def _elbow_index(values: Sequence[float]) -> int:
    """Return the index of the maximum-curvature point on a sorted desc curve."""
    n = len(values)
    if n < 3:
        return n - 1
    x = np.arange(n, dtype=float)
    y = np.asarray(values, dtype=float)
    # Normalise to [0,1] axes
    if x.max() == x.min() or y.max() == y.min():
        return n - 1
    x_n = (x - x.min()) / (x.max() - x.min())
    y_n = (y - y.min()) / (y.max() - y.min())
    # Distance from the chord to each point — the elbow is the max distance
    diff = y_n - (1 - x_n)  # chord from (0,1) to (1,0) for descending curve
    return int(np.argmax(diff))


# ---- BH FDR -------------------------------------------------------------


def _bh_threshold(p_values: Sequence[float], *, fdr: float) -> int:
    """Return how many of the sorted p-values survive Benjamini-Hochberg."""
    if not p_values:
        return 0
    sorted_ps = sorted(p_values)
    m = len(sorted_ps)
    last_pass = 0
    for i, p in enumerate(sorted_ps, start=1):
        if p <= (i / m) * fdr:
            last_pass = i
    return last_pass


# ---- main calibration ---------------------------------------------------


def calibrate_agent(
    traces: Sequence[NormalizedTrace],
    *,
    target_fdr: float = 0.05,
    target_queue_min: int = 4,
    target_queue_max: int = 30,
) -> CalibratedThresholds:
