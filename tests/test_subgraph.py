"""Tests for subgraph mining over DAG traces."""

from __future__ import annotations

from datetime import UTC, datetime

from latentspec.mining.subgraph import (
    mine_frequent_edges,
    mine_frequent_forks,
    mine_frequent_paths,
    run_subgraph_mining,
)
from latentspec.models.invariant import InvariantType
from latentspec.schemas.dag import DagTrace, EdgeKind, TraceEdge, TraceNode
from latentspec.schemas.trace import (
    AgentResponseStep,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)


def _seq_dag(idx: int, tools: list[str]) -> DagTrace:
    base = datetime.now(UTC)
    nodes = [TraceNode(id="n0", step=UserInputStep(content="hi"), started_at=base)]
    nodes.extend(
        TraceNode(id=f"n{i+1}", step=ToolCallStep(tool=tool, args={}), started_at=base)
        for i, tool in enumerate(tools)
    )
    nodes.append(
        TraceNode(
            id=f"n{len(tools)+1}",
            step=AgentResponseStep(content="done"),
            started_at=base,
        )
    )
    edges = [
        TraceEdge(source=f"n{i}", target=f"n{i+1}", kind=EdgeKind.SEQUENTIAL)
        for i in range(len(nodes) - 1)
    ]
    return DagTrace(
        trace_id=f"seq-{idx}",
        agent_id="a",
        timestamp=base,
        nodes=nodes,
        edges=edges,
        metadata=TraceMetadata(),
    )


def _fork_dag(idx: int, parent: str, children: list[str]) -> DagTrace:
    base = datetime.now(UTC)
    nodes = [TraceNode(id="n0", step=ToolCallStep(tool=parent, args={}), started_at=base)]
    nodes.extend(
        TraceNode(
            id=f"n{i+1}", step=ToolCallStep(tool=child, args={}), started_at=base
        )
        for i, child in enumerate(children)
    )
    edges = [
        TraceEdge(source="n0", target=f"n{i+1}", kind=EdgeKind.SPAWN)
        for i, _ in enumerate(children)
    ]
    return DagTrace(
        trace_id=f"fork-{idx}",
        agent_id="a",
        timestamp=base,
        nodes=nodes,
        edges=edges,
        metadata=TraceMetadata(),
    )


def test_mine_frequent_edges_recovers_pair() -> None:
    traces = [_seq_dag(i, ["auth", "db_write"]) for i in range(20)]
    candidates = mine_frequent_edges(traces, min_support=0.6)
    assert any(
        c.extra.get("tool_a") == "auth" and c.extra.get("tool_b") == "db_write"
        for c in candidates
    )


def test_mine_frequent_paths_recovers_chain() -> None:
    traces = [_seq_dag(i, ["validate", "auth", "db_write"]) for i in range(20)]
    candidates = mine_frequent_paths(traces, min_support=0.6, max_length=4)
    chain = next(
        (c for c in candidates if c.extra.get("chain") == ["validate", "auth", "db_write"]),
        None,
    )
    assert chain is not None
    assert chain.extra["pattern_length"] == 3


def test_mine_frequent_forks_recovers_concurrent_children() -> None:
    traces = [
        _fork_dag(i, "search_flights", ["check_inventory", "lookup_pricing"])
        for i in range(20)
    ]
    candidates = mine_frequent_forks(traces, min_support=0.6, min_fork_arity=2)
    fork = next(
        (c for c in candidates if c.extra.get("upstream_tool") == "search_flights"),
        None,
    )
    assert fork is not None
    assert set(fork.extra["fork_children"]) == {"check_inventory", "lookup_pricing"}
    assert fork.type == InvariantType.COMPOSITION


def test_run_subgraph_mining_combines_all_three() -> None:
    seq = [_seq_dag(i, ["auth", "db_write"]) for i in range(15)]
    forks = [
        _fork_dag(i, "search", ["check_a", "check_b"]) for i in range(15)
    ]
    result = run_subgraph_mining(seq + forks, min_support=0.4)
    assert result.edges
    assert result.forks
    # Path mining returns chain candidates; with only 2-tool seqs we expect
    # at least the pair-as-path
    assert all(c.extra.get("subgraph_mined") for c in result.all())
