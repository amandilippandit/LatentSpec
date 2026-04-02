"""PrefixSpan with closed-pattern mining for ordering invariants (§3.3).

This implementation does the real work the doc names rather than a
two-tool pair shortcut:

  1. Mine all frequent contiguous-or-not subsequences of length 2..max_len
     using PrefixSpan-style projected databases.
  2. Filter to *closed* patterns (no superpattern with the same support)
     so we don't drown the user in subsumed rules.
  3. For each closed pattern of length 2 we additionally check directional
     consistency (P(A→B) vs P(B→A)) and gap consistency (B follows A
     within `max_gap` steps in `min_directional_gap_consistency` traces).

The result is a tight set of ordering candidates with high signal: each
emitted invariant captures a real, surprise-tier rule (e.g. "validate_input
always precedes load_session, which always precedes search_flights").

Complexity: O(D × |Σ|^max_len) worst case where D = number of traces and
|Σ| = number of unique tools. Agent vocabularies are tiny (10s of tools),
max_len caps at 4 by default, and projected-DB pruning is aggressive — the
miner runs in milliseconds on the synthetic 240-trace demo.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from latentspec.models.invariant import InvariantType, Severity
from latentspec.schemas.invariant import InvariantCandidate
from latentspec.schemas.trace import NormalizedTrace, ToolCallStep


def _tool_sequence(trace: NormalizedTrace) -> list[str]:
    return [
        s.tool for s in trace.steps if isinstance(s, ToolCallStep)
    ]


# ---------------- PrefixSpan core -----------------------------------------


def _frequent_singletons(
    sequences: list[list[str]], min_count: int
) -> list[str]:
    counts: Counter[str] = Counter()
    for seq in sequences:
        # PrefixSpan support is *per-sequence*: each seq contributes ≤1 to
        # each item's count.
        for item in set(seq):
            counts[item] += 1
    return [item for item, c in counts.items() if c >= min_count]


def _project(
    sequences: list[list[str]], item: str
) -> list[list[str]]:
    """Project each sequence onto everything *strictly after* the first
    occurrence of `item`. PrefixSpan canonical projection."""
    projected: list[list[str]] = []
    for seq in sequences:
        for i, x in enumerate(seq):
            if x == item:
                projected.append(seq[i + 1 :])
                break
    return projected


def _prefixspan(
    sequences: list[list[str]],
    *,
    prefix: tuple[str, ...],
    min_count: int,
    max_len: int,
    out: dict[tuple[str, ...], int],
) -> None:
    """Recursive projected-database mining; populate `out` with patterns + supports."""
    if len(prefix) >= max_len:
        return
    # Frequent items in the projected DB
    counts: Counter[str] = Counter()
    for seq in sequences:
        for item in set(seq):
            counts[item] += 1
    for item, c in counts.items():
        if c < min_count:
            continue
        new_prefix = prefix + (item,)
        out[new_prefix] = c
        projected = _project(sequences, item)
        if projected:
            _prefixspan(
                projected,
                prefix=new_prefix,
                min_count=min_count,
                max_len=max_len,
                out=out,
            )


def _closed_patterns(
    patterns: dict[tuple[str, ...], int],
) -> dict[tuple[str, ...], int]:
    """A pattern P is *closed* iff no superpattern has the same support."""
    closed: dict[tuple[str, ...], int] = {}
    for pat, sup in patterns.items():
        is_closed = True
        for other, other_sup in patterns.items():
            if pat == other or len(other) <= len(pat):
                continue
            if other_sup != sup:
                continue
            if _is_subsequence(pat, other):
                is_closed = False
                break
        if is_closed:
            closed[pat] = sup
    return closed


def _is_subsequence(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Return True iff `a` is a (non-contiguous) subsequence of `b`."""
    it = iter(b)
    return all(item in it for item in a)


# ---------------- Public API ----------------------------------------------


def mine_sequences(
    traces: list[NormalizedTrace],
    *,
    min_support: float = 0.6,
    min_directionality: float = 0.9,
    max_pattern_length: int = 3,
    max_pair_gap: int = 5,
    min_gap_consistency: float = 0.8,
    closed_only: bool = True,
) -> list[InvariantCandidate]:
    """Discover ordering invariants from a normalized trace set.

    Args:
