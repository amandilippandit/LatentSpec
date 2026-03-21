"""Apply a canonicalisation result to a NormalizedTrace.

Used at trace ingest time so downstream mining + checking sees only
canonical tool names.
"""

from __future__ import annotations

from latentspec.canonicalization.canonicalizer import CanonicalisationResult
from latentspec.schemas.trace import NormalizedTrace, ToolCallStep


def apply_canonicalisation(
    trace: NormalizedTrace, result: CanonicalisationResult
) -> NormalizedTrace:
    """Return a copy of `trace` with `tool` field replaced by canonical name."""
    new_steps = []
    for step in trace.steps:
        if isinstance(step, ToolCallStep):
            canonical = result.canonicalise(step.tool)
            if canonical != step.tool:
                step = step.model_copy(update={"tool": canonical})
        new_steps.append(step)
    return trace.model_copy(update={"steps": new_steps})


def apply_alias_map(
    trace: NormalizedTrace, alias_map: dict[str, str]
) -> NormalizedTrace:
    """Faster path used at runtime — apply a precomputed alias map."""
    new_steps = []
    for step in trace.steps:
        if isinstance(step, ToolCallStep):
            canonical = alias_map.get(step.tool, step.tool)
            if canonical != step.tool:
                step = step.model_copy(update={"tool": canonical})
        new_steps.append(step)
    return trace.model_copy(update={"steps": new_steps})
