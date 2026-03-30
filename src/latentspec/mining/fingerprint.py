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
