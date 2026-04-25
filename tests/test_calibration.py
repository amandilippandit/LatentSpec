"""Tests for per-agent threshold calibration."""

from __future__ import annotations

from datetime import UTC, datetime

from latentspec.calibration.calibrator import calibrate_agent
from latentspec.demo import generate_traces
from latentspec.schemas.trace import (
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)


def test_calibrator_emits_per_agent_thresholds() -> None:
    traces = generate_traces(120, seed=11)
    th = calibrate_agent(traces)
    # Calibrated thresholds must stay in physically meaningful ranges
    assert 0.3 <= th.mining_min_support <= 0.85
    assert 2 <= th.mining_max_path_length <= 6
    assert 0.0 < th.anomaly_contamination < 1.0
    assert 0.6 < th.confidence_review_threshold < 1.0
    assert th.fingerprint_chi_square_threshold > 0
    assert th.drift_ph_threshold > 0
    assert th.drift_cusum_threshold > 0


def test_calibrator_distribution_summary_populated() -> None:
    traces = generate_traces(100, seed=3)
    th = calibrate_agent(traces)
    summary = th.distribution_summary
    assert summary.n_traces == 100
    assert summary.n_distinct_tools > 0
    assert summary.n_distinct_fingerprints > 0
    assert summary.median_steps_per_trace > 0
    assert len(summary.pattern_support_curve) > 0


def test_calibrator_returns_defaults_for_tiny_corpus() -> None:
    traces = generate_traces(5, seed=3)  # below the 30-trace floor
    th = calibrate_agent(traces)
    assert th.mining_min_support == 0.6
    assert th.confidence_review_threshold == 0.8


def test_higher_novelty_widens_drift_thresholds() -> None:
    """Agents with high fingerprint diversity get LARGER drift thresholds
    so PH/CUSUM doesn't fire on every new shape."""
    # Stable agent — every trace has the same shape
    stable_steps = [UserInputStep(content="x"), ToolCallStep(tool="t", args={})]
    stable = [
        NormalizedTrace(
            trace_id=f"s-{i}",
            agent_id="stable",
            timestamp=datetime.now(UTC),
            steps=stable_steps,
            metadata=TraceMetadata(),
        )
        for i in range(60)
    ]
    # Diverse agent — every trace has a different tool
    diverse = [
        NormalizedTrace(
            trace_id=f"d-{i}",
            agent_id="diverse",
            timestamp=datetime.now(UTC),
            steps=[UserInputStep(content="x"), ToolCallStep(tool=f"t{i}", args={})],
            metadata=TraceMetadata(),
        )
        for i in range(60)
    ]
    th_stable = calibrate_agent(stable)
    th_diverse = calibrate_agent(diverse)
    assert th_diverse.drift_ph_threshold > th_stable.drift_ph_threshold


def test_higher_error_rate_increases_anomaly_contamination() -> None:
    clean = [
        NormalizedTrace(
            trace_id=f"c-{i}",
            agent_id="a",
            timestamp=datetime.now(UTC),
            steps=[
                UserInputStep(content="x"),
                ToolCallStep(tool="t", args={}, result_status="success"),
            ],
            metadata=TraceMetadata(),
        )
        for i in range(60)
    ]
    noisy = [
        NormalizedTrace(
            trace_id=f"n-{i}",
            agent_id="a",
            timestamp=datetime.now(UTC),
            steps=[
                UserInputStep(content="x"),
                ToolCallStep(tool="t", args={}, result_status="error" if i % 5 == 0 else "success"),
            ],
            metadata=TraceMetadata(),
        )
        for i in range(60)
    ]
    th_clean = calibrate_agent(clean)
    th_noisy = calibrate_agent(noisy)
    assert th_noisy.anomaly_contamination > th_clean.anomaly_contamination
