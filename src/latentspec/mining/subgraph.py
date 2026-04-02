"""Subgraph mining over DAG traces.

PrefixSpan is a sequence-mining algorithm — it assumes traces are linear
chains. For DAG traces (branching, parallel, looping) we need a real graph
miner. Full gSpan is overkill for our domain (agent traces are small —
typically < 50 nodes, < 100 edges, max-degree < 6); we ship a focused
algorithm tuned to that shape:

  - **Frequent edges** — count `(source_tool, edge_kind, target_tool)`
    triples across the trace corpus. High-support triples become
    composition / sequential ordering invariants.
  - **Frequent rooted paths** — extend frequent edges into rooted paths
    of length up to `max_path_length`. Same projection-based recursion
    PrefixSpan uses, but over graph successor sets rather than sequence
    suffixes.
  - **Frequent fork patterns** — for each node with branching out-degree,
    record the multiset of children. Frequent multisets become
    "this tool always spawns these N tools concurrently" rules.

The output candidates plug into the existing `cross_validate +
formalize + checker dispatch` pipeline. Subgraph candidates emit as
ORDERING (paths) or COMPOSITION (forks) so the existing checkers
evaluate them without modification.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from latentspec.models.invariant import InvariantType, Severity
from latentspec.schemas.dag import DagTrace, EdgeKind, TraceNode
from latentspec.schemas.invariant import InvariantCandidate
from latentspec.schemas.trace import ToolCallStep

log = logging.getLogger(__name__)


# ---- helpers -------------------------------------------------------------


def _tool_of(node: TraceNode) -> str | None:
    if isinstance(node.step, ToolCallStep):
        return node.step.tool
    return None


def _build_index(trace: DagTrace) -> tuple[dict[str, TraceNode], dict[str, list[str]]]:
    by_id = {n.id: n for n in trace.nodes}
    fwd: dict[str, list[str]] = defaultdict(list)
    for e in trace.edges:
        if e.kind == EdgeKind.LOOP:
            continue
        if e.source in by_id and e.target in by_id:
            fwd[e.source].append(e.target)
    return by_id, fwd


# ---- frequent edges ------------------------------------------------------


def mine_frequent_edges(
    traces: Sequence[DagTrace],
    *,
    min_support: float = 0.5,
) -> list[InvariantCandidate]:
    """Mine `(parent_tool, child_tool)` triples with high cross-trace support.

    A triple `(A, kind, B)` is frequent when ≥ `min_support` of the trace
    corpus contains an edge of that kind from a `tool=A` node to a
    `tool=B` node. Each emitted candidate is an ORDERING invariant.
    """
    if not traces:
        return []
    n = len(traces)
    min_count = max(1, int(min_support * n))

    edge_counts: Counter[tuple[str, str, str]] = Counter()
    edge_evidence: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    for trace in traces:
        by_id, _ = _build_index(trace)
        seen_in_trace: set[tuple[str, str, str]] = set()
        for e in trace.edges:
            if e.kind == EdgeKind.LOOP:
                continue
            src_tool = _tool_of(by_id.get(e.source)) if e.source in by_id else None
            tgt_tool = _tool_of(by_id.get(e.target)) if e.target in by_id else None
            if not src_tool or not tgt_tool:
                continue
            key = (src_tool, e.kind.value, tgt_tool)
            if key in seen_in_trace:
                continue
            seen_in_trace.add(key)
            edge_counts[key] += 1
            edge_evidence[key].append(trace.trace_id)

    out: list[InvariantCandidate] = []
    for (src, kind, tgt), count in edge_counts.items():
        if count < min_count:
            continue
        support = count / n
        out.append(
            InvariantCandidate(
                type=InvariantType.ORDERING,
                description=f"`{src}` is followed by `{tgt}` ({kind} edge)",
                formal_rule=(
                    f"forall trace: edge({src}, {kind}, {tgt}) in trace.dag"
                ),
                evidence_trace_ids=edge_evidence[(src, kind, tgt)][:50],
                support=round(support, 4),
                consistency=round(support, 4),
                severity=Severity.HIGH,
                discovered_by="statistical",
                extra={
                    "tool_a": src,
                    "tool_b": tgt,
                    "edge_kind": kind,
                    "co_occurrence": count,
                    "subgraph_mined": True,
                },
            )
        )
    return out


# ---- frequent rooted paths ----------------------------------------------


def _enumerate_paths(
    trace: DagTrace, *, max_length: int
) -> list[tuple[str, ...]]:
    """All rooted paths up to `max_length` of tool-name labels."""
    by_id, fwd = _build_index(trace)
    paths: list[tuple[str, ...]] = []

    def dfs(node_id: str, prefix: tuple[str, ...], remaining: int) -> None:
        node = by_id.get(node_id)
        if node is None:
            return
        tool = _tool_of(node)
        new_prefix = prefix
        if tool:
            new_prefix = prefix + (tool,)
            if len(new_prefix) >= 2:
                paths.append(new_prefix)
        if remaining <= 0:
            return
        for nxt in fwd.get(node_id, []):
            dfs(nxt, new_prefix, remaining - 1)

    for n in trace.nodes:
        dfs(n.id, (), max_length)
    return paths

