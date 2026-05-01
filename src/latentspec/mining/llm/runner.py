"""Track B driver — chunks traces into batches and aggregates candidates.

Per §3.2, batches are 50–100 traces. We deduplicate near-identical
descriptions across batches by simple normalization so the same rule
discovered in two adjacent batches doesn't appear twice.
"""

from __future__ import annotations

import asyncio
import logging
import re

from latentspec.config import get_settings
from latentspec.mining.llm.claude import ClaudeMiner, LLMMiningError
from latentspec.schemas.invariant import InvariantCandidate
from latentspec.schemas.trace import NormalizedTrace

log = logging.getLogger(__name__)


def _dedup_key(c: InvariantCandidate) -> str:
    desc = c.description.lower()
    desc = re.sub(r"[^a-z0-9_ ]+", "", desc)
    desc = re.sub(r"\s+", " ", desc).strip()
    return f"{c.type.value}::{desc}"


def _merge(a: InvariantCandidate, b: InvariantCandidate) -> InvariantCandidate:
    """Combine two near-identical candidates by averaging support/consistency."""
    return a.model_copy(
        update={
            "support": round((a.support + b.support) / 2, 4),
            "consistency": round((a.consistency + b.consistency) / 2, 4),
            "evidence_trace_ids": list(
                dict.fromkeys([*a.evidence_trace_ids, *b.evidence_trace_ids])
            )[:50],
        }
    )


async def run_llm_track(
    traces: list[NormalizedTrace],
    *,
    miner: ClaudeMiner | None = None,
    batch_size: int | None = None,
) -> list[InvariantCandidate]:
    """Run Claude over the trace batch (or sub-batches) and aggregate candidates."""
    if not traces:
        return []
    settings = get_settings()
    if not settings.anthropic_api_key:
        log.warning("ANTHROPIC_API_KEY not set — skipping Track B")
        return []

    miner = miner or ClaudeMiner()
    bs = batch_size or settings.mining_batch_size

    batches = [traces[i : i + bs] for i in range(0, len(traces), bs)]

    try:
        results = await asyncio.gather(
            *(miner.mine_batch(batch) for batch in batches), return_exceptions=True
        )
    except LLMMiningError as e:
        log.warning("Track B aborted: %s", e)
        return []

    aggregated: dict[str, InvariantCandidate] = {}
    for r in results:
        if isinstance(r, Exception):
            log.warning("LLM batch failed: %s", r)
            continue
        for cand in r:
            key = _dedup_key(cand)
            if key in aggregated:
                aggregated[key] = _merge(aggregated[key], cand)
            else:
                aggregated[key] = cand
    return list(aggregated.values())
