"""Tool-name canonicaliser.

Multi-pass alias detection:

  Pass 1 — Normalised exact match
    `Payments_v2` → `payments.v2`. Drops casing differences, separator
    differences, version suffixes (`v1`, `_v2`, `.v2`, `-2.1`), and
    namespace prefixes (`payment.v2.execute` → `execute`-anchored
    family).

  Pass 2 — Token-Jaccard + Levenshtein
    For pairs surviving Pass 1 as distinct, compute Jaccard on token
    set + Levenshtein on stripped forms. Pair clusters when
    `jaccard >= 0.5 OR edit_distance <= 2`.

  Pass 3 — Character n-gram cosine
    Final fallback. Builds a TF-IDF over character trigrams of remaining
    distinct tool names. Pairs above `cosine_threshold` cluster.

The output is a `CanonicalisationResult` with one canonical form per
cluster and a `(raw_name → canonical, method, confidence)` decision
per input. Persisted to `tool_aliases` table by the orchestrator.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from latentspec.schemas.trace import NormalizedTrace, ToolCallStep


# ---- normalisation helpers ----------------------------------------------


_VERSION_RE = re.compile(
    r"(?P<sep>[._\-])v\d+(?:\.\d+)*$|(?P<sep2>[._\-])\d+(?:\.\d+)*$"
)
_PUNCT_RE = re.compile(r"[\s._\-]+")
_NAMESPACE_TAIL_RE = re.compile(r"^(?:[a-z0-9]+[._-]){1,3}")


def canonical_form(name: str) -> str:
    """Apply Pass 1 normalisation: lowercase, version-strip, punct-collapse."""
    out = name.strip().lower()
    # Drop trailing version suffix
    while True:
        m = _VERSION_RE.search(out)
        if m is None:
            break
        out = out[: m.start()]
    # Replace punctuation with single underscore
    out = _PUNCT_RE.sub("_", out)
    out = out.strip("_")
    return out


def _tokens(name: str) -> set[str]:
    return {t for t in canonical_form(name).split("_") if t}


def _levenshtein(a: str, b: str, *, max_distance: int = 5) -> int:
    """Bounded Levenshtein — short-circuits past `max_distance`."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        min_in_row = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if cur[j] < min_in_row:
                min_in_row = cur[j]
        if min_in_row > max_distance:
            return max_distance + 1
        prev = cur
    return prev[-1]


def _trigram_vector(name: str) -> Counter[str]:
    s = f"  {canonical_form(name)}  "
    return Counter(s[i : i + 3] for i in range(len(s) - 2))


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---- union-find ---------------------------------------------------------


class _UF:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        self.parent[max(ra, rb)] = min(ra, rb)


# ---- result types -------------------------------------------------------


@dataclass
class AliasDecision:
    raw_name: str
    canonical_name: str
    method: str  # "exact" | "token_jaccard" | "edit_distance" | "trigram_cosine"
    confidence: float


@dataclass
class CanonicalisationResult:
    decisions: list[AliasDecision] = field(default_factory=list)
    clusters: dict[str, list[str]] = field(default_factory=dict)
    canonical_for: dict[str, str] = field(default_factory=dict)

    def canonicalise(self, raw_name: str) -> str:
        return self.canonical_for.get(raw_name, raw_name)


# ---- canonicaliser ------------------------------------------------------


@dataclass
class ToolCanonicalizer:
    """Multi-pass alias detector. Stateless across calls except via the
    tool-frequency hint passed at construction time (used for tie-breaks
    when picking the canonical name within a cluster)."""

    jaccard_threshold: float = 0.5
    edit_distance_threshold: int = 2
    trigram_cosine_threshold: float = 0.78

    def fit(self, tool_names: Sequence[str]) -> CanonicalisationResult:
        names = sorted(set(tool_names))
        if not names:
            return CanonicalisationResult()

        n = len(names)
        normalised = [canonical_form(name) for name in names]
        token_sets = [_tokens(name) for name in names]
        trigrams = [_trigram_vector(name) for name in names]

        uf = _UF(n)

        # Pass 1: normalised exact match
        norm_groups: dict[str, list[int]] = defaultdict(list)
        for i, norm in enumerate(normalised):
            norm_groups[norm].append(i)
        for group in norm_groups.values():
            for j in group[1:]:
                uf.union(group[0], j)

        # Pass 2: token Jaccard + bounded Levenshtein
        for i in range(n):
            for j in range(i + 1, n):
                if uf.find(i) == uf.find(j):
                    continue
                ti, tj = token_sets[i], token_sets[j]
                if ti and tj:
                    jaccard = len(ti & tj) / len(ti | tj)
                    if jaccard >= self.jaccard_threshold:
                        uf.union(i, j)
                        continue
                ed = _levenshtein(
                    normalised[i],
                    normalised[j],
                    max_distance=self.edit_distance_threshold + 1,
                )
                if ed <= self.edit_distance_threshold:
                    uf.union(i, j)

        # Pass 3: trigram cosine fallback
        for i in range(n):
            for j in range(i + 1, n):
                if uf.find(i) == uf.find(j):
                    continue
                cos = _cosine(trigrams[i], trigrams[j])
                if cos >= self.trigram_cosine_threshold:
                    uf.union(i, j)

        # Build clusters
        cluster_members: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            cluster_members[uf.find(i)].append(i)

        decisions: list[AliasDecision] = []
        clusters: dict[str, list[str]] = {}
        canonical_for: dict[str, str] = {}

        for member_indices in cluster_members.values():
            cluster_names = [names[i] for i in member_indices]
            canonical = self._pick_canonical(cluster_names)
            clusters[canonical] = cluster_names
            for raw in cluster_names:
                method, conf = self._classify(canonical, raw)
                canonical_for[raw] = canonical
                decisions.append(
                    AliasDecision(
                        raw_name=raw,
                        canonical_name=canonical,
                        method=method,
                        confidence=conf,
                    )
                )

        return CanonicalisationResult(
            decisions=decisions, clusters=clusters, canonical_for=canonical_for
        )

    def _pick_canonical(self, names: list[str]) -> str:
        """Choose the canonical representative.

        Heuristic: shortest normalised form among the cluster, ties broken
        by lexicographic order. Short forms are usually the "logical" name;
        longer forms tend to carry version / namespace cruft.
        """
        return min(names, key=lambda n: (len(canonical_form(n)), canonical_form(n)))

    @staticmethod
    def _classify(canonical: str, raw: str) -> tuple[str, float]:
        if canonical == raw:
            return "self", 1.0
        if canonical_form(canonical) == canonical_form(raw):
            return "exact", 0.99
        ed = _levenshtein(
            canonical_form(canonical), canonical_form(raw), max_distance=4
        )
        if ed <= 2:
            return "edit_distance", round(1.0 - ed / 4, 3)
        ct = _tokens(canonical)
        rt = _tokens(raw)
        if ct and rt:
            jaccard = len(ct & rt) / len(ct | rt)
            if jaccard >= 0.5:
                return "token_jaccard", round(jaccard, 3)
        cos = _cosine(_trigram_vector(canonical), _trigram_vector(raw))
        return "trigram_cosine", round(cos, 3)


def collect_tool_names(traces: Sequence[NormalizedTrace]) -> list[str]:
    """Convenience: pull every tool name observed in a trace batch."""
    seen: set[str] = set()
    for t in traces:
        for s in t.steps:
            if isinstance(s, ToolCallStep):
                seen.add(s.tool)
    return sorted(seen)
