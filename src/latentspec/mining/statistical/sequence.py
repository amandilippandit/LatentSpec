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
        traces: §3.2 normalized traces.
        min_support: minimum fraction of traces in which a pattern must
            occur to be considered (PrefixSpan support).
        min_directionality: for length-2 patterns, P(A→B | A,B both present)
            must exceed this. Equal to 1.0 = always-A-before-B.
        max_pattern_length: longest subsequence considered. Defaults to 3
            (length-3 patterns capture chains like A→B→C without exploding
            the search space).
        max_pair_gap: a length-2 pair (A, B) is "tight" only when B occurs
            within `max_pair_gap` steps of A. Patterns that fail the gap
            check are still emitted but with a lower severity.
        min_gap_consistency: in what fraction of traces must B fall within
            `max_pair_gap` of A for the pattern to be marked HIGH severity.
        closed_only: drop subsumed patterns. Set False for debugging.
    """
    if not traces:
        return []

    n = len(traces)
    min_count = max(1, int(min_support * n))
    sequences = [_tool_sequence(t) for t in traces]

    raw: dict[tuple[str, ...], int] = {}
    _prefixspan(
        sequences,
        prefix=(),
        min_count=min_count,
        max_len=max_pattern_length,
        out=raw,
    )
    raw = {p: c for p, c in raw.items() if len(p) >= 2}

    patterns = _closed_patterns(raw) if closed_only else raw

    # Pair-level directional consistency (length-2 patterns only)
    forward_count: Counter[tuple[str, str]] = Counter()
    cooccur: Counter[frozenset[str]] = Counter()
    gap_consistent: dict[tuple[str, str], int] = defaultdict(int)
    pattern_evidence: dict[tuple[str, ...], list[str]] = defaultdict(list)

    for trace, seq in zip(traces, sequences, strict=True):
        unique = set(seq)
        for a in unique:
            for b in unique:
                if a < b:
                    cooccur[frozenset({a, b})] += 1
        # forward direction
        first_idx: dict[str, int] = {}
        for idx, t in enumerate(seq):
            if t not in first_idx:
                first_idx[t] = idx
        for a, ai in first_idx.items():
            for b, bi in first_idx.items():
                if a == b:
                    continue
                if ai < bi:
                    forward_count[(a, b)] += 1
                    if (bi - ai) <= max_pair_gap:
                        gap_consistent[(a, b)] += 1
        # pattern evidence
        for pat in patterns:
            if _occurs_in_order(pat, seq):
                pattern_evidence[pat].append(trace.trace_id)

    candidates: list[InvariantCandidate] = []
    for pat, sup_count in sorted(patterns.items(), key=lambda kv: (-len(kv[0]), -kv[1])):
        support = sup_count / n
        evidence = pattern_evidence.get(pat, [])[:50]
        if len(pat) == 2:
            a, b = pat
            co = cooccur[frozenset({a, b})]
            if co == 0:
                continue
            forward = forward_count[(a, b)]
            backward = forward_count[(b, a)]
            denom = forward + backward
            if denom == 0:
                continue
            directionality = forward / denom
            if directionality < min_directionality:
                continue
            gap_consistency = gap_consistent[(a, b)] / max(1, forward)
            severity = (
                Severity.HIGH if gap_consistency >= min_gap_consistency else Severity.MEDIUM
            )
            description = f"The agent always calls `{a}` before calling `{b}`"
            formal_rule = (
                f"forall trace: if contains(trace, {b}) "
                f"then exists(step_before({b}, {a}))"
            )
            candidates.append(
                InvariantCandidate(
                    type=InvariantType.ORDERING,
                    description=description,
                    formal_rule=formal_rule,
                    evidence_trace_ids=evidence,
                    support=round(support, 4),
                    consistency=round(directionality, 4),
                    severity=severity,
                    discovered_by="statistical",
                    extra={
                        "tool_a": a,
                        "tool_b": b,
                        "co_occurrence": co,
                        "directionality": round(directionality, 4),
                        "gap_consistency": round(gap_consistency, 4),
                        "pattern_length": 2,
                    },
                )
            )
        else:
            # Length-3+ chain — emit as a chained ordering invariant. The
            # checker for these uses a sliding-window index match.
            description = (
                "The agent always calls "
                + " then ".join(f"`{t}`" for t in pat)
            )
            formal_rule = "forall trace: contains_subsequence(trace, " + ", ".join(pat) + ")"
            candidates.append(
                InvariantCandidate(
                    type=InvariantType.ORDERING,
                    description=description,
                    formal_rule=formal_rule,
                    evidence_trace_ids=evidence,
                    support=round(support, 4),
                    consistency=round(support, 4),
                    severity=Severity.MEDIUM,
                    discovered_by="statistical",
                    extra={
                        "tool_a": pat[0],
                        "tool_b": pat[-1],
                        "chain": list(pat),
                        "pattern_length": len(pat),
                    },
                )
            )

    return candidates


def _occurs_in_order(pattern: tuple[str, ...], seq: Iterable[str]) -> bool:
    it = iter(seq)
    return all(item in it for item in pattern)
