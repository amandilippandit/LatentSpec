"""Conditional invariant mining via mutual information (§3.3).

The previous implementation used hand-rolled support × confidence scoring
with a hardcoded stop-word list. That works for the demo but conflates
correlation strength with rule selectivity, and over-emits weak rules.

This rewrite uses **mutual information** between binary features (keyword
present in user_input?) and binary outcomes (was tool T invoked?) as the
ranking criterion. Mutual information is the information-theoretic
quantity that measures how much knowing one feature reduces uncertainty
about the other. It's the right scoring function for "does this keyword
predict this tool call?" because:

  - I(K; T) = 0 iff K and T are independent (good for filtering).
  - I(K; T) is naturally bounded by entropy, so we can compare across
    keyword-tool pairs on the same scale.
  - Combined with a chi-square significance test we control false
    positives that high-MI low-support pairs would otherwise create.

We also drop the static stop-word list in favour of an in-corpus
document-frequency filter: tokens appearing in ≥ 80% of trace user_inputs
are treated as too common to be discriminative (the standard high-DF cut).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from latentspec.models.invariant import InvariantType, Severity
from latentspec.schemas.invariant import InvariantCandidate
from latentspec.schemas.trace import NormalizedTrace, ToolCallStep, UserInputStep


_TOKEN_RE = re.compile(r"[a-z][a-z0-9_]{2,}")


@dataclass(frozen=True)
class _Counts:
    """2x2 contingency for (keyword present, tool invoked)."""

    n11: int  # both present
    n10: int  # keyword present, tool absent
    n01: int  # keyword absent, tool present
    n00: int  # both absent

    @property
    def total(self) -> int:
        return self.n11 + self.n10 + self.n01 + self.n00


def _trace_tokens(trace: NormalizedTrace) -> set[str]:
    out: set[str] = set()
    for step in trace.steps:
        if isinstance(step, UserInputStep):
            for tok in _TOKEN_RE.findall(step.content.lower()):
                out.add(tok)
    return out


def _trace_tools(trace: NormalizedTrace) -> set[str]:
    return {s.tool for s in trace.steps if isinstance(s, ToolCallStep)}


def _filter_high_df(
    token_doc_count: Counter[str], n_docs: int, *, max_df: float
) -> set[str]:
    """Drop tokens appearing in >= max_df fraction of documents."""
    if n_docs == 0:
        return set()
    cutoff = int(max_df * n_docs)
    return {tok for tok, df in token_doc_count.items() if df >= cutoff}


def _mutual_information(c: _Counts) -> float:
    """I(K; T) over the 2x2 contingency, in bits."""
    n = c.total
    if n == 0:
        return 0.0
    p_k = (c.n11 + c.n10) / n
    p_t = (c.n11 + c.n01) / n
    if p_k in (0.0, 1.0) or p_t in (0.0, 1.0):
        return 0.0

    mi = 0.0
    cells = [
        (c.n11, p_k * p_t),
        (c.n10, p_k * (1 - p_t)),
        (c.n01, (1 - p_k) * p_t),
        (c.n00, (1 - p_k) * (1 - p_t)),
    ]
    for cnt, denom_p in cells:
        if cnt == 0 or denom_p == 0:
            continue
        p_obs = cnt / n
        mi += p_obs * math.log2(p_obs / denom_p)
    return max(0.0, mi)


def _chi_square(c: _Counts) -> float:
    """Pearson's chi-square for the 2x2 contingency.

    Used as a significance filter — pairs with high MI but small counts
    have unstable estimates; chi-square's df=1 critical value 3.84 (95% CI)
    rejects most of them.
    """
    n = c.total
    if n == 0:
        return 0.0
    row1 = c.n11 + c.n10
    row2 = c.n01 + c.n00
    col1 = c.n11 + c.n01
    col2 = c.n10 + c.n00
    if row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
        return 0.0

    expected = [
        row1 * col1 / n, row1 * col2 / n,
        row2 * col1 / n, row2 * col2 / n,
    ]
    observed = [c.n11, c.n10, c.n01, c.n00]
    chi = 0.0
    for o, e in zip(observed, expected, strict=True):
        if e > 0:
            chi += (o - e) ** 2 / e
    return chi


def mine_associations(
    traces: list[NormalizedTrace],
    *,
    min_support: float = 0.05,
    min_mi_bits: float = 0.05,
    chi_square_threshold: float = 6.63,  # df=1, p < 0.01
    min_keyword_traces: int = 10,
    min_lift: float = 0.2,
    max_df: float = 0.8,
) -> list[InvariantCandidate]:
    """Mine `keyword K → tool T` rules ranked by mutual information.

    Filters:
      - keyword DF in [min_keyword_traces, max_df * N] (skip too-rare/common)
      - MI(K; T) >= `min_mi_bits` (uniform discriminative-ness floor)
      - chi-square >= `chi_square_threshold` (statistical significance)
      - lift = P(T|K) - P(T) >= `min_lift` (rule effect size)
    """
    if not traces:
        return []

    n = len(traces)
    keyword_traces: dict[str, set[int]] = defaultdict(set)
    tool_traces: dict[str, set[int]] = defaultdict(set)

    for idx, trace in enumerate(traces):
        for tok in _trace_tokens(trace):
            keyword_traces[tok].add(idx)
        for tool in _trace_tools(trace):
            tool_traces[tool].add(idx)

    # Document-frequency filter (replaces the static stop-word list).
    token_df = Counter({tok: len(s) for tok, s in keyword_traces.items()})
    too_common = _filter_high_df(token_df, n, max_df=max_df)

    candidates: list[tuple[float, InvariantCandidate]] = []
    for kw, kw_set in keyword_traces.items():
        if kw in too_common:
            continue
        if len(kw_set) < min_keyword_traces:
            continue

        for tool, tool_set in tool_traces.items():
            both = len(kw_set & tool_set)
            kw_only = len(kw_set) - both
            tool_only = len(tool_set) - both
            neither = n - both - kw_only - tool_only
            if neither < 0:  # rare numerical edge case; skip
                continue

            counts = _Counts(both, kw_only, tool_only, neither)
            support = both / n
            if support < min_support:
                continue

            mi = _mutual_information(counts)
            if mi < min_mi_bits:
                continue

            chi = _chi_square(counts)
            if chi < chi_square_threshold:
                continue

            p_t = len(tool_set) / n
            p_t_given_k = both / max(1, len(kw_set))
            lift = p_t_given_k - p_t
            if lift < min_lift:
                continue

            evidence_ids = [traces[i].trace_id for i in list(kw_set & tool_set)[:50]]
            description = (
                f"When the user input mentions '{kw}', the agent uses tool `{tool}`"
            )
            candidate = InvariantCandidate(
                type=InvariantType.CONDITIONAL,
                description=description,
                formal_rule=(
                    f"forall trace: contains_keyword(trace.user_input, '{kw}') "
                    f"-> exists(step in trace.tool_calls where step.tool == '{tool}')"
                ),
                evidence_trace_ids=evidence_ids,
                support=round(support, 4),
                consistency=round(p_t_given_k, 4),
                severity=Severity.MEDIUM,
                discovered_by="statistical",
                extra={
                    "keyword": kw,
                    "tool": tool,
                    "keyword_traces": len(kw_set),
                    "lift": round(lift, 4),
                    "mutual_information_bits": round(mi, 4),
                    "chi_square": round(chi, 3),
                },
            )
            candidates.append((mi, candidate))

    # Rank by MI; deduplicate per (keyword, tool) — the highest-MI win.
    candidates.sort(key=lambda x: -x[0])
    seen: set[tuple[str, str]] = set()
    out: list[InvariantCandidate] = []
    for _mi, cand in candidates:
        key = (cand.extra["keyword"], cand.extra["tool"])
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out
