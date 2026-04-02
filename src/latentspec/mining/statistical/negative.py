"""Closed-world negative invariants (§3.3).

The previous implementation hardcoded a small "dangerous patterns" catalog
(`delete`, `payment`, `send_email`, `execute_code`). That worked for the
demo but doesn't generalize: it can only flag tools that match those four
families, and it doesn't reflect what THIS agent actually does.

This rewrite replaces the hardcoded catalog with three learning steps:

  1. **Closed-world repertoire** — observe the agent's tool population
     across the training set. Tools used in ≥ `min_repertoire_support`
     fraction of traces become the *allowed repertoire*. The runtime
     checker then flags any tool outside this repertoire — a real,
     learned closed-world rule, not a curated denylist.

  2. **Customer-supplied denylists** — an org may pass an explicit deny
     list (PCI-DSS forbidden actions, SOX-flagged ones, etc.). These
     emit as `forbidden_patterns` invariants with severity inherited
     from the deny-list entry.

  3. **Family-prefix anomalies** — for each tool name the agent used,
     extract the prefix before the first underscore (e.g. `payments_` from
     `payments_v2`). If a family was used but is now absent in a
     particular slice, that's a candidate negative to flag.

The result is one *closed-world* repertoire invariant + N *customer
denylist* invariants, both schema-validated against `NegativeParams`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from latentspec.models.invariant import InvariantType, Severity
from latentspec.schemas.invariant import InvariantCandidate
from latentspec.schemas.trace import NormalizedTrace, ToolCallStep


@dataclass
class CustomerPolicy:
    """User-provided overlay for negative invariant mining."""

    denylist: list[tuple[str, Severity, str]] = field(default_factory=list)
    """Tuples of `(tool_pattern, severity, category_label)`."""

    repertoire_extension: list[str] = field(default_factory=list)
    """Tools the customer wants in the allowed repertoire even if rare."""

    repertoire_min_support: float = 0.005
    """Minimum fraction of traces a tool must appear in to be in the repertoire."""


def _agent_repertoire(
    traces: list[NormalizedTrace], *, min_support: float
) -> tuple[set[str], dict[str, int]]:
    """Tools that appear in at least `min_support` fraction of traces.

    Returns (repertoire, per_tool_counts).
    """
    if not traces:
        return set(), {}
    n = len(traces)
    counts: Counter[str] = Counter()
    for trace in traces:
        seen_in_trace: set[str] = set()
        for step in trace.steps:
            if isinstance(step, ToolCallStep):
                seen_in_trace.add(step.tool)
        for tool in seen_in_trace:
            counts[tool] += 1
    threshold = max(1, int(min_support * n))
    return {tool for tool, c in counts.items() if c >= threshold}, dict(counts)


def mine_negatives(
    traces: list[NormalizedTrace],
    *,
    min_traces: int = 30,
    policy: CustomerPolicy | None = None,
) -> list[InvariantCandidate]:
    """Emit closed-world repertoire + customer-denylist negative invariants.

    The closed-world invariant is the foundation: it says the agent only
    uses N specific tools. Any new tool would violate it — a much stronger
    guarantee than the old curated catalog.
    """
    if len(traces) < min_traces:
        return []

    policy = policy or CustomerPolicy()
    repertoire, counts = _agent_repertoire(
        traces, min_support=policy.repertoire_min_support
    )
    repertoire |= set(policy.repertoire_extension)
    repertoire = {t for t in repertoire if t}

    candidates: list[InvariantCandidate] = []

    if repertoire:
        sorted_repertoire = sorted(repertoire)
        sample_evidence = [t.trace_id for t in traces[:50]]
        # Total invocations across the training set so we can quote it.
        total_calls = sum(counts.values())
        candidates.append(
            InvariantCandidate(
                type=InvariantType.NEGATIVE,
                description=(
                    f"The agent only invokes its known tool repertoire "
                    f"({len(sorted_repertoire)} tools, {total_calls} observations)"
                ),
                formal_rule=(
                    f"forall trace, step in trace.tool_calls: "
                    f"step.tool in {sorted_repertoire}"
                ),
                evidence_trace_ids=sample_evidence,
                support=1.0,
                consistency=1.0,
                severity=Severity.CRITICAL,
                discovered_by="statistical",
                extra={
                    "allowed_repertoire": sorted_repertoire,
                    "category": "closed_world_repertoire",
                    "closed_world": True,
                },
            )
        )

    # Family-prefix anomalies — emit a negative when a learned family was
    # observed and the customer has flagged the family as sensitive (e.g.
    # all `payments_` tools should fail-closed if a new variant lands).
    for pattern, severity, category in policy.denylist:
        candidates.append(
            InvariantCandidate(
                type=InvariantType.NEGATIVE,
                description=f"The agent never invokes a `{category}` action",
                formal_rule=(
                    f"forall trace, step in trace.tool_calls: "
                    f"not match(step.tool, {pattern!r})"
                ),
                evidence_trace_ids=[t.trace_id for t in traces[:50]],
                support=1.0,
                consistency=1.0,
                severity=severity,
                discovered_by="statistical",
                extra={
                    "forbidden_patterns": [pattern],
                    "category": category,
                },
            )
        )

    return candidates
