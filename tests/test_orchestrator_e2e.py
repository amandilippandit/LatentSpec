"""End-to-end mining test using the synthetic booking-agent.

Validates the §9 week-1 milestone in-memory: 200+ traces from a sample agent
must produce ≥10 meaningful invariants spanning ≥3 §3.3 types. We run with
LLM disabled so the test does not require an Anthropic API key.
"""

from __future__ import annotations

import os
import uuid

import pytest

# Make sure Track B is skipped in the unit-test environment
os.environ["ANTHROPIC_API_KEY"] = ""

from latentspec.demo import generate_traces  # noqa: E402
from latentspec.mining.orchestrator import mine_invariants  # noqa: E402


@pytest.mark.asyncio
async def test_week1_milestone_offline() -> None:
    traces = generate_traces(240, seed=7)
    result = await mine_invariants(
        agent_id=uuid.uuid4(),
        traces=traces,
        session=None,
        persist=False,
    )
    n = len(result.invariants)
    types = set(result.by_type)
    # Track A alone (no LLM) must be sufficient to clear the gate.
    assert n >= 10, f"only got {n} invariants; expected ≥10"
    assert len(types) >= 3, f"only got {len(types)} types: {types}"
    # Spot-check that ordering, statistical and negative all appear
    assert "ordering" in types
    assert "statistical" in types
    assert "negative" in types
