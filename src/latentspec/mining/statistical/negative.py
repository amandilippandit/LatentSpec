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
