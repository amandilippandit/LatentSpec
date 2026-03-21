"""Per-agent threshold calibration.

Rather than ship one set of hardcoded thresholds for every agent on
earth, the calibrator inspects a representative trace sample for each
agent and picks the thresholds that maximise discriminative signal
without flooding the user with false-positive rules.

Five quantities calibrated per agent:

  - `mining_min_support` — at what fraction of traces is a sequence
    pattern worth surfacing? Bootstrapped from the per-pattern frequency
    distribution: pick the knee point where adding more rules doesn't
    add signal (elbow detection on the support curve).

  - `association_min_mi_bits` — minimum mutual information for a
    keyword→tool rule. Calibrated from the empirical distribution of MI
    values in the corpus + a multiple-testing correction so the false
    discovery rate stays below `target_fdr`.

  - `confidence_review_threshold` — the §3.4 0.8 default is too low for
    high-volume agents (floods the review queue) and too high for
    low-volume ones (auto-promotes nothing). Calibrated to land the
    expected pending queue size in `[target_queue_min, target_queue_max]`.

  - `fingerprint_chi_square_threshold` — the χ² critical value
    depends on degrees of freedom = n_distinct_fingerprints - 1.
    Calibrated by computing the actual baseline distribution's effective
    df and looking up the right critical value.

  - `drift_ph_threshold` — Page-Hinkley fires sooner on noisy streams
    than on clean ones. Calibrated to the smoothed pass-rate variance.

Persisted to `calibration_results` table per agent. Mining and the
streaming detector consult the calibrated values when present.
"""

from latentspec.calibration.calibrator import (
    CalibratedThresholds,
    DistributionSummary,
    calibrate_agent,
)

__all__ = ["CalibratedThresholds", "DistributionSummary", "calibrate_agent"]
