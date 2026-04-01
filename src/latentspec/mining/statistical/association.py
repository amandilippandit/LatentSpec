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

