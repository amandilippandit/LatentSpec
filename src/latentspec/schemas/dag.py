"""Typed DAG trace schema (extension §3.2 + §3.3 Composition).

The §3.2 `NormalizedTrace.steps` field is a flat ordered list. That works
for linear agent runs, but breaks down for:

  - Multi-agent orchestrations where step S1 (agent A) feeds S2 and S3
    (agents B and C running concurrently), then both feed S4.
  - LangGraph / StateGraph agents with conditional edges and loops.
  - ReAct agents whose `thought → action → observation → thought` cycles
    are best modeled as a graph where back-edges encode loops.
  - Concurrent tool fan-out (`asyncio.gather` over N tools).

This module adds an alternative DAG view (`DagTrace`) alongside the
existing linear schema. A linear `NormalizedTrace` converts trivially
into a DAG (one chain of edges); a DAG-shaped trace keeps full
parallelism / branching / merge information.

The runtime checker, the streaming detector, and the Z3 compiler all
*linearize* the DAG via topological sort when needed, so existing
checkers keep working unchanged. Subgraph mining (`mining/subgraph.py`)
then operates on the native DAG.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from latentspec.schemas.trace import NormalizedTrace, TraceMetadata, TraceStep


class EdgeKind(str, enum.Enum):
    SEQUENTIAL = "sequential"  # explicit ordering between two steps
    DATA = "data"              # output of source consumed by target
    SPAWN = "spawn"            # source forked target (parallel branch)
    JOIN = "join"              # source joined into target (parallel merge)
    LOOP = "loop"              # back-edge — target is a prior node
    CONDITIONAL = "conditional"  # one of N branches taken


class TraceNode(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    step: TraceStep
    started_at: datetime | None = None
    ended_at: datetime | None = None
    agent_role: str | None = None  # multi-agent: which agent owns this step

    @property
    def duration_ms(self) -> int | None:
        if self.started_at and self.ended_at:
            return max(0, int((self.ended_at - self.started_at).total_seconds() * 1000))
        return None


class TraceEdge(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str  # node id
    target: str  # node id
    kind: EdgeKind = EdgeKind.SEQUENTIAL
    condition: str | None = None  # for CONDITIONAL edges
    metadata: dict[str, Any] = Field(default_factory=dict)


class DagTrace(BaseModel):
    model_config = ConfigDict(extra="allow")

    trace_id: str
    agent_id: str
    timestamp: datetime
    nodes: list[TraceNode]
    edges: list[TraceEdge] = Field(default_factory=list)
    metadata: TraceMetadata = Field(default_factory=TraceMetadata)

    @classmethod
    def from_linear(cls, trace: NormalizedTrace) -> "DagTrace":
        """Lift a linear NormalizedTrace into a chain-shaped DAG."""
        nodes = [
            TraceNode(id=f"n{i}", step=step) for i, step in enumerate(trace.steps)
        ]
        edges = [
            TraceEdge(source=f"n{i}", target=f"n{i+1}", kind=EdgeKind.SEQUENTIAL)
            for i in range(len(trace.steps) - 1)
        ]
        return cls(
            trace_id=trace.trace_id,
            agent_id=trace.agent_id,
            timestamp=trace.timestamp,
            nodes=nodes,
            edges=edges,
            metadata=trace.metadata,
        )

    def to_linear(self) -> NormalizedTrace:
        """Topologically sort and project to a linear NormalizedTrace.

        Concurrent / merged nodes interleave by start time. Loops collapse
        to a single visit per node. The result is suitable for passing to
        any of the existing rule-based checkers, the Z3 verifier, etc.
        """
        ordered = self._topological_sort()
        steps = [node.step for node in ordered]
        return NormalizedTrace(
            trace_id=self.trace_id,
            agent_id=self.agent_id,
            timestamp=self.timestamp,
            steps=steps,
            metadata=self.metadata,
        )

    def _topological_sort(self) -> list[TraceNode]:
        """Kahn's algorithm with start-time tiebreaker."""
        node_by_id = {n.id: n for n in self.nodes}
        # Forward + reverse adjacency, ignoring back-edges to keep DAG-ness
        forward: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        in_deg: dict[str, int] = {n.id: 0 for n in self.nodes}
        for e in self.edges:
            if e.kind == EdgeKind.LOOP:
                continue
            if e.source not in forward or e.target not in in_deg:
                continue
            forward[e.source].append(e.target)
            in_deg[e.target] += 1

        # Initial frontier: nodes with no inbound, sorted by start time
        ready = [
            n for n in self.nodes if in_deg.get(n.id, 0) == 0
        ]
        ready.sort(key=lambda n: n.started_at or self.timestamp)

        out: list[TraceNode] = []
        while ready:
            node = ready.pop(0)
            out.append(node)
            for nxt_id in forward.get(node.id, []):
                in_deg[nxt_id] -= 1
                if in_deg[nxt_id] == 0:
                    ready.append(node_by_id[nxt_id])
                    ready.sort(key=lambda n: n.started_at or self.timestamp)

        # Append any remaining (cycle-only) nodes in insertion order
        seen = {n.id for n in out}
        for n in self.nodes:
            if n.id not in seen:
                out.append(n)
        return out

    @property
    def has_branching(self) -> bool:
        out_degree: dict[str, int] = {}
        for e in self.edges:
            out_degree[e.source] = out_degree.get(e.source, 0) + 1
        return any(d > 1 for d in out_degree.values())

    @property
    def has_loops(self) -> bool:
        return any(e.kind == EdgeKind.LOOP for e in self.edges)
