"""Per-cluster mining orchestrator.

Clusters traces first, then runs the existing mining pipeline once per
cluster. Each emitted invariant is stamped with its `cluster_id` so the
runtime checker can route incoming traces to the right rule subset.

This unlocks open-ended agents that the global orchestrator can't mine —
the hard combinatorial problem ("what does this agent do?") becomes N
small problems ("what does this agent do *in this workflow*?").
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field

from latentspec.config import get_settings
from latentspec.mining.clustering import (
    WorkflowClustering,
    cluster_workflows,
    split_by_cluster,
)
from latentspec.mining.confidence import cross_validate
from latentspec.mining.formalization import formalize
from latentspec.mining.llm.runner import run_llm_track
from latentspec.mining.statistical.runner import run_statistical_track
from latentspec.models.invariant import InvariantStatus
from latentspec.schemas.invariant import InvariantCandidate, MinedInvariant
from latentspec.schemas.trace import NormalizedTrace

log = logging.getLogger(__name__)


@dataclass
class ClusterMiningResult:
    cluster_id: int
    n_traces: int
    invariants: list[MinedInvariant] = field(default_factory=list)
    candidates_statistical: int = 0
    candidates_llm: int = 0


@dataclass
class ClusteredMiningResult:
    n_traces: int
    n_clusters: int
    silhouette: float
    cluster_sizes: dict[int, int]
    per_cluster: list[ClusterMiningResult] = field(default_factory=list)
    invariants: list[MinedInvariant] = field(default_factory=list)


async def mine_per_cluster(
    *,
    agent_id: uuid.UUID,
    traces: list[NormalizedTrace],
    k_min: int = 2,
    k_max: int = 12,
    min_traces_per_cluster: int = 16,
) -> tuple[ClusteredMiningResult, WorkflowClustering]:
    """Cluster `traces`, mine each cluster, return the union of invariants.

    Each invariant carries `cluster_id` in its params so the runtime
    checker only applies it to traces routed to that cluster.
    """
    if not traces:
        raise ValueError("mine_per_cluster: empty trace set")

    settings = get_settings()
    clustering = cluster_workflows(
