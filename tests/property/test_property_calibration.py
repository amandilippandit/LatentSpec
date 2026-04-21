"""Property-based tests for the threshold calibrator.

Invariants:

  - thresholds always lie in physically meaningful ranges
  - calibrating an empty / tiny corpus returns defaults (not crashes)
  - distribution_summary fields all >= 0
"""

from __future__ import annotations

from hypothesis import given, settings, HealthCheck, strategies as st

from latentspec.calibration.calibrator import calibrate_agent
from tests.property.strategies import normalized_trace


@given(traces=st.lists(normalized_trace(max_steps=12), min_size=0, max_size=80))
@settings(max_examples=30, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
def test_calibrator_thresholds_in_physical_range(traces) -> None:
    th = calibrate_agent(traces)

    # All thresholds bounded in their physical range
    assert 0.0 < th.mining_min_support <= 1.0
    assert 0.0 < th.mining_min_directionality <= 1.0
    assert th.mining_max_path_length >= 2
    assert th.association_min_mi_bits >= 0.0
    assert 0.0 <= th.association_min_lift <= 1.0
    assert th.association_min_keyword_traces >= 1
    assert 0.0 < th.statistical_p_target <= 100
    assert 0.0 < th.anomaly_contamination < 1.0
    assert 0.0 <= th.confidence_reject_threshold <= 1.0
    assert 0.0 <= th.confidence_review_threshold <= 1.0
    assert th.confidence_reject_threshold <= th.confidence_review_threshold
    assert th.fingerprint_chi_square_threshold > 0
    assert th.drift_ph_threshold > 0
    assert th.drift_cusum_threshold > 0


@given(traces=st.lists(normalized_trace(max_steps=12), min_size=0, max_size=80))
@settings(max_examples=20, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
def test_distribution_summary_fields_non_negative(traces) -> None:
    th = calibrate_agent(traces)
    s = th.distribution_summary
    assert s.n_traces >= 0
    assert s.n_distinct_tools >= 0
    assert s.n_distinct_fingerprints >= 0
    assert s.median_steps_per_trace >= 0
    assert s.p99_steps_per_trace >= 0
    assert s.median_tool_calls_per_trace >= 0
