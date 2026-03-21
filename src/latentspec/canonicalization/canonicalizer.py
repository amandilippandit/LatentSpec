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
