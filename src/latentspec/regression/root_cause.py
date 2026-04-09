"""§4.3 root-cause hint generator.

Given a failed invariant + a passing baseline trace + a failing candidate
trace, ask Claude to explain the diff in one paragraph. The output is
plain English, no formal-methods jargon, designed to land in the
violation-analysis dashboard view.

Falls back to a deterministic templated explanation when no API key is set.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from latentspec.config import get_settings
from latentspec.regression.batch import InvariantBatchSummary
from latentspec.schemas.trace import NormalizedTrace

log = logging.getLogger(__name__)


ROOT_CAUSE_SYSTEM = """\
You are LatentSpec's root-cause analyst. You read one behavioral rule
that broke between two trace samples (a passing baseline and a failing
candidate) and explain in plain English what changed.

Your output is ONE short paragraph (3 sentences max). Plain English.
No formal-methods jargon. No bullet points. End with a concrete next
step the developer should investigate.
"""


@dataclass
class RootCauseHint:
    invariant_id: str
    description: str
    hypothesis: str
    affected_segments: list[str]


def _fallback_hypothesis(summary: InvariantBatchSummary) -> str:
    pct_fail = round(summary.fail_rate * 100)
    return (
        f"The rule '{summary.description}' is now violated in {pct_fail}% of "
        f"applicable traces. Inspect the most recent diff for changes that "
        f"affect this behavior — likely candidates: prompt edits, tool removal, "
        f"or new conditional branches that bypass the rule."
    )


async def _generate_one(
    summary: InvariantBatchSummary,
    baseline_trace: NormalizedTrace | None,
    failing_trace: NormalizedTrace | None,
) -> RootCauseHint:
    settings = get_settings()
    affected_segments = _collect_segments([failing_trace] if failing_trace else [])

    if not settings.anthropic_api_key:
        return RootCauseHint(
            invariant_id=summary.invariant_id,
            description=summary.description,
            hypothesis=_fallback_hypothesis(summary),
            affected_segments=affected_segments,
        )

    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    payload = {
        "rule": summary.description,
        "baseline_pass_rate": None,
        "candidate_pass_rate": summary.pass_rate,
        "candidate_fail_rate": summary.fail_rate,
        "baseline_trace": _summarize_trace(baseline_trace),
        "failing_trace": _summarize_trace(failing_trace),
    }
    try:
        resp = await client.messages.create(
