"""SQLAlchemy ORM models implementing the §8.1 core entities + week-3 auth + extensions.

Five product tables (§8.1): agents, traces, invariants, violations, mining_runs.
Three auth tables (§9 week-3 day-5): organizations, users, api_keys.
Nine extension tables: tool_aliases, agent_versions, fingerprint_baselines,
  cluster_centroids, drift_state, sessions, synthetic_review_queue,
  mining_jobs, calibration_results.
"""

from latentspec.models.agent import Agent
from latentspec.models.auth import ApiKey, Organization, PricingTier, User
from latentspec.models.extensions import (
    AgentVersion,
    CalibrationResult,
    ClusterCentroid,
    DriftState,
    FingerprintBaseline,
    JobKind,
    JobStatus,
    MiningJob,
    ReviewDecision,
    Session,
    SyntheticReviewItem,
    ToolAlias,
)
from latentspec.models.invariant import Invariant, InvariantStatus, InvariantType, Severity
from latentspec.models.mining_run import MiningRun, MiningRunStatus
from latentspec.models.trace import Trace, TraceStatus
from latentspec.models.violation import Violation

__all__ = [
    "Agent",
    "AgentVersion",
    "ApiKey",
    "CalibrationResult",
    "ClusterCentroid",
    "DriftState",
    "FingerprintBaseline",
    "Invariant",
    "InvariantStatus",
    "InvariantType",
    "JobKind",
    "JobStatus",
    "MiningJob",
    "MiningRun",
    "MiningRunStatus",
    "Organization",
    "PricingTier",
    "ReviewDecision",
    "Session",
    "Severity",
    "SyntheticReviewItem",
    "ToolAlias",
    "Trace",
    "TraceStatus",
    "User",
    "Violation",
]
