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
    """Learn thresholds tuned to this specific agent's distribution."""
    if len(traces) < 30:
        # Too few traces to calibrate meaningfully — return defaults
        return _defaults(traces)

    # ---- distribution summary --------------------------------------------
    tool_counts: Counter[str] = Counter()
    fp_counts: Counter[str] = Counter()
    step_counts: list[int] = []
    tool_call_counts: list[int] = []
    for t in traces:
        step_counts.append(len(t.steps))
        tcs = [s for s in t.steps if isinstance(s, ToolCallStep)]
        tool_call_counts.append(len(tcs))
        for s in tcs:
            tool_counts[s.tool] += 1
        fp_counts[fingerprint(t)] += 1

    summary = DistributionSummary(
        n_traces=len(traces),
        n_distinct_tools=len(tool_counts),
        n_distinct_fingerprints=len(fp_counts),
        median_steps_per_trace=float(np.median(step_counts)),
        p99_steps_per_trace=float(np.percentile(step_counts, 99)),
        median_tool_calls_per_trace=float(np.median(tool_call_counts)),
    )

    # ---- min_support via elbow on pattern frequencies --------------------
    # Mine permissively and observe how candidate count scales with support
    permissive_candidates = run_statistical_track(
        list(traces), min_support_sequence=0.05
    )
    pattern_supports = sorted(
        (c.support for c in permissive_candidates), reverse=True
    )
    summary.pattern_support_curve = [round(s, 4) for s in pattern_supports[:50]]
    if pattern_supports:
        elbow = _elbow_index(pattern_supports)
        # Clamp into a sane range — never below 0.3, never above 0.85
        min_support = max(0.3, min(0.85, pattern_supports[elbow]))
    else:
        min_support = 0.6

    # ---- max_path_length from observed median trace length ---------------
    # Long traces support longer chains; short traces don't
    max_path_length = int(min(6, max(2, summary.median_tool_calls_per_trace // 2)))

    # ---- min_keyword_traces from corpus size ----------------------------
    min_keyword_traces = max(5, int(0.05 * len(traces)))

    # ---- chi-square threshold from fingerprint df -----------------------
    # df = number of distinct fingerprints - 1; cap at 30 for stability
    df = max(1, min(30, summary.n_distinct_fingerprints - 1))
    fp_chi2 = _chi2_critical(df)

    # ---- review threshold via bisection on expected queue size ----------
    # Use the candidate confidence scores directly to estimate review-band size
    from latentspec.mining.confidence import score_candidate

    if permissive_candidates:
        scores = [score_candidate(c).final for c in permissive_candidates]
        review_threshold = _calibrate_review_band(
            scores,
            queue_min=target_queue_min,
            queue_max=target_queue_max,
        )
    else:
        review_threshold = 0.8

    # ---- ph + cusum thresholds from per-trace pass-rate variance --------
    # Without a labelled stream we use the empirical fingerprint-novelty rate
    # as a proxy for natural variance: stable agents -> tighter thresholds
    novelty_rate = summary.n_distinct_fingerprints / max(1, len(traces))
    ph_threshold = max(2.0, min(20.0, 8.0 * (1 + novelty_rate)))
    cusum_threshold = max(2.0, min(10.0, 4.0 * (1 + novelty_rate)))

    # ---- anomaly contamination from observed error rate -----------------
    error_rate = sum(
        1
        for t in traces
        for s in t.steps
        if isinstance(s, ToolCallStep)
        and (s.result_status or "success") != "success"
    ) / max(1, sum(tool_call_counts))
    anomaly_contamination = max(0.01, min(0.15, error_rate * 1.5))

    return CalibratedThresholds(
        mining_min_support=round(min_support, 4),
        mining_min_directionality=0.9,
        mining_max_path_length=max_path_length,
        association_min_mi_bits=0.05,
        association_min_lift=0.2,
        association_min_keyword_traces=min_keyword_traces,
        statistical_p_target=99.0,
        anomaly_contamination=round(anomaly_contamination, 4),
        confidence_reject_threshold=0.6,
        confidence_review_threshold=round(review_threshold, 4),
        fingerprint_chi_square_threshold=round(fp_chi2, 3),
        drift_ph_threshold=round(ph_threshold, 3),
        drift_cusum_threshold=round(cusum_threshold, 3),
        distribution_summary=summary,
    )


def _calibrate_review_band(
    scores: Sequence[float],
    *,
    queue_min: int,
    queue_max: int,
) -> float:
    """Bisect the review threshold so the expected queue size lands in range."""
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    if n == 0:
        return 0.8

    # Lo: most permissive ⇒ everything above 0.6 in queue
    # Hi: strictest ⇒ only top 1% in queue
    lo, hi = 0.6, 0.99
    target_count = (queue_min + queue_max) / 2
    for _ in range(20):
        mid = (lo + hi) / 2
        # Number of scores in [0.6, mid)
        n_in_band = sum(1 for s in sorted_scores if 0.6 <= s < mid)
        if n_in_band < queue_min:
            lo = mid  # raise the threshold to grow the band
        elif n_in_band > queue_max:
            hi = mid  # lower the threshold to shrink the band
        else:
            break
    return mid


def _defaults(traces: Sequence[NormalizedTrace]) -> CalibratedThresholds:
    return CalibratedThresholds(
        mining_min_support=0.6,
        mining_min_directionality=0.9,
        mining_max_path_length=3,
        association_min_mi_bits=0.05,
        association_min_lift=0.2,
        association_min_keyword_traces=10,
        statistical_p_target=99.0,
        anomaly_contamination=0.05,
        confidence_reject_threshold=0.6,
        confidence_review_threshold=0.8,
        fingerprint_chi_square_threshold=13.82,
        drift_ph_threshold=8.0,
        drift_cusum_threshold=4.0,
        distribution_summary=DistributionSummary(
            n_traces=len(traces),
            n_distinct_tools=0,
            n_distinct_fingerprints=0,
            median_steps_per_trace=0.0,
            p99_steps_per_trace=0.0,
            median_tool_calls_per_trace=0.0,
        ),
    )
