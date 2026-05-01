"""Tests for the typed DAG trace schema and linear↔DAG conversion."""

from __future__ import annotations

from datetime import UTC, datetime

from latentspec.schemas.dag import DagTrace, EdgeKind, TraceEdge, TraceNode
from latentspec.schemas.trace import (
    AgentResponseStep,
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)


def _linear_trace() -> NormalizedTrace:
    return NormalizedTrace(
        trace_id="t-linear",
        agent_id="a",
        timestamp=datetime.now(UTC),
        steps=[
            UserInputStep(content="hi"),
            ToolCallStep(tool="auth", args={}),
            ToolCallStep(tool="db_write", args={}),
            AgentResponseStep(content="done"),
        ],
        metadata=TraceMetadata(),
    )


def test_linear_to_dag_round_trip() -> None:
    linear = _linear_trace()
    dag = DagTrace.from_linear(linear)
    assert len(dag.nodes) == len(linear.steps)
    assert len(dag.edges) == len(linear.steps) - 1
    assert all(e.kind == EdgeKind.SEQUENTIAL for e in dag.edges)
    relinearized = dag.to_linear()
    assert [s.type for s in relinearized.steps] == [s.type for s in linear.steps]


def test_branching_dag_topo_sort_preserves_order() -> None:
    # Diamond: n0 -> n1, n0 -> n2, n1 -> n3, n2 -> n3
    base = datetime.now(UTC)
    nodes = [
        TraceNode(id="n0", step=UserInputStep(content="go"), started_at=base),
        TraceNode(id="n1", step=ToolCallStep(tool="left", args={}), started_at=base),
        TraceNode(id="n2", step=ToolCallStep(tool="right", args={}), started_at=base),
        TraceNode(id="n3", step=AgentResponseStep(content="merged"), started_at=base),
    ]
    edges = [
        TraceEdge(source="n0", target="n1", kind=EdgeKind.SPAWN),
        TraceEdge(source="n0", target="n2", kind=EdgeKind.SPAWN),
        TraceEdge(source="n1", target="n3", kind=EdgeKind.JOIN),
        TraceEdge(source="n2", target="n3", kind=EdgeKind.JOIN),
    ]
    dag = DagTrace(
        trace_id="diamond",
        agent_id="a",
        timestamp=base,
        nodes=nodes,
        edges=edges,
        metadata=TraceMetadata(),
    )
    assert dag.has_branching
    assert not dag.has_loops
    linearized = dag.to_linear()
    # Order: user_input first, response last; left/right interleave between.
    types = [s.type.value for s in linearized.steps]
    assert types[0] == "user_input"
    assert types[-1] == "agent_response"


def test_loop_edges_dont_break_topo_sort() -> None:
    base = datetime.now(UTC)
    nodes = [
        TraceNode(id="n0", step=UserInputStep(content="go"), started_at=base),
        TraceNode(id="n1", step=ToolCallStep(tool="t1", args={}), started_at=base),
        TraceNode(id="n2", step=AgentResponseStep(content="done"), started_at=base),
    ]
    edges = [
        TraceEdge(source="n0", target="n1", kind=EdgeKind.SEQUENTIAL),
        TraceEdge(source="n1", target="n0", kind=EdgeKind.LOOP),  # back-edge
        TraceEdge(source="n1", target="n2", kind=EdgeKind.SEQUENTIAL),
    ]
    dag = DagTrace(
        trace_id="loop",
        agent_id="a",
        timestamp=base,
        nodes=nodes,
        edges=edges,
        metadata=TraceMetadata(),
    )
    assert dag.has_loops
    linear = dag.to_linear()
    # Each node visited once despite the loop edge
    assert len(linear.steps) == 3
