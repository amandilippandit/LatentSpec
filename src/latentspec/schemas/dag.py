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
