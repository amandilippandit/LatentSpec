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
            model=settings.anthropic_model,
            max_tokens=320,
            temperature=0.2,
            system=ROOT_CAUSE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                }
            ],
        )
        text = "".join(getattr(b, "text", "") or "" for b in resp.content).strip()
        if not text:
            text = _fallback_hypothesis(summary)
    except Exception as e:
        log.warning("root-cause LLM call failed: %s", e)
        text = _fallback_hypothesis(summary)

    return RootCauseHint(
        invariant_id=summary.invariant_id,
        description=summary.description,
        hypothesis=text,
        affected_segments=affected_segments,
    )


async def generate_root_cause_hints(
    failures: list[InvariantBatchSummary],
    baseline_traces: list[NormalizedTrace],
    candidate_traces: list[NormalizedTrace],
    *,
    concurrency: int = 4,
) -> list[RootCauseHint]:
    """Generate one hint per failed invariant."""
    if not failures:
        return []

    bl_by_id = {t.trace_id: t for t in baseline_traces}
    cd_by_id = {t.trace_id: t for t in candidate_traces}

    sem = asyncio.Semaphore(concurrency)

    async def _bounded(summary: InvariantBatchSummary) -> RootCauseHint:
        async with sem:
            failing = (
                cd_by_id.get(summary.sample_failure_traces[0])
                if summary.sample_failure_traces
                else None
            )
            baseline = next(iter(bl_by_id.values())) if bl_by_id else None
            return await _generate_one(summary, baseline, failing)

    return await asyncio.gather(*(_bounded(f) for f in failures))


def _summarize_trace(trace: NormalizedTrace | None) -> dict | None:
    if trace is None:
        return None
    return {
        "trace_id": trace.trace_id,
        "metadata": trace.metadata.model_dump(exclude_none=True),
        "steps": [
            {
                k: v
                for k, v in s.model_dump().items()
                if k in {"type", "tool", "args", "latency_ms", "result_status", "content"}
                and v is not None
            }
            for s in trace.steps
        ],
    }


def _collect_segments(traces: list[NormalizedTrace | None]) -> list[str]:
    out: list[str] = []
    for t in traces:
        if t is None:
            continue
        seg = t.metadata.user_segment
        if seg and seg not in out:
            out.append(seg)
    return out
