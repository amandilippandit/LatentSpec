"""Behavioral regression detection (§4).

Two checker classes per §3.3:
  - rule-based  (sub-100ms, deterministic) — ordering, conditional, negative,
                 statistical, state, composition, tool_selection
  - LLM-as-judge (sub-2s, probabilistic)   — output_format only

The dispatcher maps invariant.type → checker so adding a new type means
adding one checker module and one registry entry.
"""

from latentspec.checking.base import (
    CheckResult,
    Checker,
    CheckerError,
    CheckOutcome,
    InvariantSpec,
    ViolationDetails,
)
from latentspec.checking.dispatch import dispatch, get_checker, register
from latentspec.checking.runner import check_trace, check_traces
from latentspec.checking.z3_checker import Z3Checker

__all__ = [
    "CheckOutcome",
    "CheckResult",
    "Checker",
    "CheckerError",
    "InvariantSpec",
    "ViolationDetails",
    "Z3Checker",
    "check_trace",
    "check_traces",
    "dispatch",
    "get_checker",
    "register",
]
