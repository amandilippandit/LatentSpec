"""Behavioral trace fingerprints + distribution drift.

A trace's *fingerprint* is a stable hash of its structural shape — the
ordered sequence of `(step_type, tool_name)` pairs. Two traces with the
same fingerprint go through the same coarse motions; two with different
fingerprints don't. Fingerprints catch the simplest behavioral pattern
violation: "this trace's shape never appeared before". They're the right
tool for simple agents (1-2 tools) where structural mining produces
nothing because everything trivially repeats.

Also exposed: `FingerprintDistribution`, a counter with KL-divergence and
chi-square drift detection so we alert when the production fingerprint mix
shifts away from the training-set baseline.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from latentspec.schemas.trace import (
    AgentResponseStep,
    AgentThoughtStep,
    NormalizedTrace,
    SystemStep,
    ToolCallStep,
    UserInputStep,
)


# ---- canonical-form computation ------------------------------------------


def _step_token(step) -> str:
    if isinstance(step, ToolCallStep):
        return f"tool:{step.tool}"
    if isinstance(step, UserInputStep):
        return "user"
    if isinstance(step, AgentResponseStep):
        return "response"
    if isinstance(step, AgentThoughtStep):
        return "thought"
    if isinstance(step, SystemStep):
        return "system"
    return "unknown"


def canonical_shape(trace: NormalizedTrace) -> str:
    """The literal sequence of step tokens, joined by `|`."""
    return "|".join(_step_token(s) for s in trace.steps)


def fingerprint(trace: NormalizedTrace, *, length: int = 16) -> str:
    """Stable 16-hex-char SHA-256 prefix of the canonical shape."""
    canonical = canonical_shape(trace)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


# ---- per-rule distribution drift -----------------------------------------


@dataclass
class FingerprintDistribution:
    """Tracks the fingerprint frequency mix for one agent.

    `update_baseline()` freezes the current counts as the reference.
    `score()` returns a tuple `(kl_divergence, chi_square)` between the
    current counts and the baseline. `is_drifting()` applies the
    chi-square critical value to decide.
    """

    counts: Counter[str] = field(default_factory=Counter)
    baseline: Counter[str] | None = None
    n_observed: int = 0
    chi_square_threshold: float = 13.82  # df ~ 5 reasonable default; configurable

    def add(self, fp: str) -> None:
        self.counts[fp] += 1
        self.n_observed += 1

    def add_trace(self, trace: NormalizedTrace) -> str:
        fp = fingerprint(trace)
        self.add(fp)
        return fp

    def update_baseline(self) -> None:
        """Freeze the current counts as the reference distribution."""
        self.baseline = Counter(self.counts)

    def score(self) -> tuple[float, float]:
        """Return (KL(current || baseline), chi^2(observed vs expected))."""
        if not self.baseline or sum(self.baseline.values()) == 0:
            return 0.0, 0.0
        bl_total = sum(self.baseline.values())
        cur_total = sum(self.counts.values())
        if cur_total == 0:
            return 0.0, 0.0

        # All keys we've seen in either distribution; smooth missing cells.
        all_keys = set(self.baseline) | set(self.counts)
        kl = 0.0
        chi = 0.0
        smooth = 0.5  # Laplace-style for unseen fingerprints
        for key in all_keys:
            p_obs = (self.counts.get(key, 0) + smooth) / (cur_total + smooth * len(all_keys))
            p_ref = (self.baseline.get(key, 0) + smooth) / (bl_total + smooth * len(all_keys))
            if p_obs > 0 and p_ref > 0:
                kl += p_obs * math.log2(p_obs / p_ref)
            expected = p_ref * cur_total
            observed = self.counts.get(key, 0)
            if expected > 0:
                chi += (observed - expected) ** 2 / expected
        return max(0.0, kl), max(0.0, chi)

    def is_drifting(self) -> bool:
        if self.baseline is None or self.n_observed < 50:
            return False
        _kl, chi = self.score()
        return chi > self.chi_square_threshold

    def novel_fingerprints(self) -> set[str]:
        """Fingerprints in current but absent from baseline — first-seen shapes."""
        if self.baseline is None:
            return set()
        return set(self.counts) - set(self.baseline)

    def reset_observation_window(self) -> None:
        self.counts = Counter()
        self.n_observed = 0


def fingerprint_set(traces: Iterable[NormalizedTrace]) -> Counter[str]:
    """One-shot: compute the fingerprint multiset of a trace batch."""
    out: Counter[str] = Counter()
    for trace in traces:
        out[fingerprint(trace)] += 1
    return out
