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
        traces,
        k_min=k_min,
        k_max=k_max,
        min_traces_per_cluster=min_traces_per_cluster,
    )
    by_cluster = split_by_cluster(traces, clustering.labels)
    log.info(
        "clustered %d traces into k=%d (silhouette=%.3f, sizes=%s)",
        len(traces), clustering.k, clustering.silhouette, clustering.cluster_sizes,
    )

    cluster_tasks: list[tuple[int, asyncio.Task]] = []

    async def _mine_one(cluster_id: int, cluster_traces: list[NormalizedTrace]):
        statistical_task = asyncio.create_task(
            asyncio.to_thread(
                run_statistical_track,
                cluster_traces,
                min_support_sequence=settings.mining_min_support,
            )
        )
        llm_task = asyncio.create_task(run_llm_track(cluster_traces))
        statistical, llm = await asyncio.gather(statistical_task, llm_task)
        merged = cross_validate(statistical, llm)

        invariants: list[MinedInvariant] = []
        for cand in merged:
            inv = formalize(
                cand,
                reject_threshold=settings.confidence_reject_threshold,
                review_threshold=settings.confidence_review_threshold,
            )
            if inv is None or inv.status == InvariantStatus.REJECTED:
                continue
            # Stamp the cluster_id into params so the checker can route
            inv.params = {**inv.params, "cluster_id": cluster_id}
            invariants.append(inv)
        return ClusterMiningResult(
            cluster_id=cluster_id,
            n_traces=len(cluster_traces),
            invariants=invariants,
            candidates_statistical=len(statistical),
            candidates_llm=len(llm),
        )

    results = await asyncio.gather(
        *(_mine_one(cid, ts) for cid, ts in by_cluster.items())
    )

    all_invariants: list[MinedInvariant] = []
    for r in results:
        all_invariants.extend(r.invariants)
    all_invariants.sort(key=lambda i: -i.confidence)

    return (
        ClusteredMiningResult(
            n_traces=len(traces),
            n_clusters=clustering.k,
            silhouette=clustering.silhouette,
            cluster_sizes=clustering.cluster_sizes,
            per_cluster=results,
            invariants=all_invariants,
        ),
        clustering,
    )


def route_trace_to_cluster(
    clustering: WorkflowClustering, trace: NormalizedTrace
) -> int:
    """Predict which cluster an incoming trace belongs to."""
    return int(clustering.predict([trace])[0])
