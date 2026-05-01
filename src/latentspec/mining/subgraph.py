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


def mine_frequent_paths(
    traces: Sequence[DagTrace],
    *,
    min_support: float = 0.5,
    max_length: int = 4,
) -> list[InvariantCandidate]:
    """Discover frequent rooted DAG paths."""
    if not traces:
        return []
    n = len(traces)
    min_count = max(1, int(min_support * n))

    path_counts: Counter[tuple[str, ...]] = Counter()
    evidence: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for trace in traces:
        # A path can occur multiple times in one trace; use set to count
        # per-trace presence, matching PrefixSpan's per-sequence support.
        unique: set[tuple[str, ...]] = set(_enumerate_paths(trace, max_length=max_length))
        for p in unique:
            path_counts[p] += 1
            evidence[p].append(trace.trace_id)

    out: list[InvariantCandidate] = []
    for path, count in path_counts.items():
        if count < min_count or len(path) < 2:
            continue
        support = count / n
        if len(path) == 2:
            description = f"The agent always reaches `{path[1]}` from `{path[0]}` in the DAG"
        else:
            description = "The agent always traces the path " + " → ".join(
                f"`{t}`" for t in path
            )
        out.append(
            InvariantCandidate(
                type=InvariantType.ORDERING,
                description=description,
                formal_rule=(
                    f"forall trace: path_exists(trace.dag, {list(path)})"
                ),
                evidence_trace_ids=evidence[path][:50],
                support=round(support, 4),
                consistency=round(support, 4),
                severity=Severity.MEDIUM,
                discovered_by="statistical",
                extra={
                    "tool_a": path[0],
                    "tool_b": path[-1],
                    "chain": list(path),
                    "pattern_length": len(path),
                    "subgraph_mined": True,
                },
            )
        )
    return out


# ---- frequent fork patterns ---------------------------------------------


def mine_frequent_forks(
    traces: Sequence[DagTrace],
    *,
    min_support: float = 0.5,
    min_fork_arity: int = 2,
) -> list[InvariantCandidate]:
    """Each parent_tool with out-degree >= 2 contributes a (parent_tool,
    sorted children) signature. Frequent signatures emit as composition
    invariants — "tool A always spawns these N children concurrently"."""
    if not traces:
        return []
    n = len(traces)
    min_count = max(1, int(min_support * n))

    fork_counts: Counter[tuple[str, tuple[str, ...]]] = Counter()
    evidence: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)

    for trace in traces:
        by_id, fwd = _build_index(trace)
        seen: set[tuple[str, tuple[str, ...]]] = set()
        for src_id, targets in fwd.items():
            if len(targets) < min_fork_arity:
                continue
            src_tool = _tool_of(by_id.get(src_id)) if src_id in by_id else None
            if not src_tool:
                continue
            child_tools = tuple(
                sorted(
                    t for t in (_tool_of(by_id.get(tid)) for tid in targets) if t
                )
            )
            if len(child_tools) < min_fork_arity:
                continue
            key = (src_tool, child_tools)
            if key in seen:
                continue
            seen.add(key)
            fork_counts[key] += 1
            evidence[key].append(trace.trace_id)

    out: list[InvariantCandidate] = []
    for (parent, children), count in fork_counts.items():
        if count < min_count:
            continue
        support = count / n
        children_str = ", ".join(f"`{c}`" for c in children)
        out.append(
            InvariantCandidate(
                type=InvariantType.COMPOSITION,
                description=f"`{parent}` always spawns concurrent calls to {children_str}",
                formal_rule=(
                    f"forall trace, src in trace.dag where src.tool == '{parent}': "
                    f"set(out_neighbors(src).tool) == {list(children)}"
                ),
                evidence_trace_ids=evidence[(parent, children)][:50],
                support=round(support, 4),
                consistency=round(support, 4),
                severity=Severity.MEDIUM,
                discovered_by="statistical",
                extra={
                    "upstream_tool": parent,
                    "downstream_tool": children[0],
                    "fork_children": list(children),
                    "subgraph_mined": True,
                },
            )
        )
    return out


# ---- driver --------------------------------------------------------------


@dataclass
class SubgraphMiningResult:
    edges: list[InvariantCandidate] = field(default_factory=list)
    paths: list[InvariantCandidate] = field(default_factory=list)
    forks: list[InvariantCandidate] = field(default_factory=list)

    def all(self) -> list[InvariantCandidate]:
        return [*self.edges, *self.paths, *self.forks]


def run_subgraph_mining(
    traces: Sequence[DagTrace],
    *,
    min_support: float = 0.5,
    max_path_length: int = 4,
) -> SubgraphMiningResult:
    return SubgraphMiningResult(
        edges=mine_frequent_edges(traces, min_support=min_support),
        paths=mine_frequent_paths(
            traces, min_support=min_support, max_length=max_path_length
        ),
        forks=mine_frequent_forks(traces, min_support=min_support),
    )
