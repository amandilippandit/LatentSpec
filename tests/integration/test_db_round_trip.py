"""Round-trip every model through SQLite.

Catches: ORM mapping errors, FK constraint violations, JSON serialisation,
async session lifecycle.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from latentspec.models import (
    Agent,
    AgentVersion,
    CalibrationResult,
    ClusterCentroid,
    DriftState,
    FingerprintBaseline,
    JobKind,
    JobStatus,
    MiningJob,
    ReviewDecision,
    Session,
    SyntheticReviewItem,
    ToolAlias,
    Trace,
    TraceStatus,
)


pytestmark = pytest.mark.asyncio


async def test_round_trip_agent(db_session, seeded_agent) -> None:
    fetched = await db_session.get(Agent, seeded_agent.id)
    assert fetched is not None
    assert fetched.name == "integration-test-agent"


async def test_round_trip_trace(db_session, seeded_agent) -> None:
    trace = Trace(
        id=uuid.uuid4(),
        agent_id=seeded_agent.id,
        trace_data={"steps": [{"type": "user_input", "content": "hi"}]},
        version_tag="v1",
        session_id="s-1",
        user_id="u-1",
        cluster_id=2,
        fingerprint="abc12345",
        started_at=datetime.now(UTC),
        ended_at=datetime.now(UTC),
        step_count=1,
        status=TraceStatus.SUCCESS,
    )
    db_session.add(trace)
    await db_session.commit()

    fetched = await db_session.get(Trace, trace.id)
    assert fetched is not None
    assert fetched.session_id == "s-1"
    assert fetched.cluster_id == 2
    assert fetched.fingerprint == "abc12345"
    assert fetched.trace_data["steps"][0]["content"] == "hi"


async def test_round_trip_tool_alias(db_session, seeded_agent) -> None:
    alias = ToolAlias(
        id=uuid.uuid4(),
        agent_id=seeded_agent.id,
        raw_name="payments_v1",
        canonical_name="payments",
        confidence=0.95,
        method="exact",
    )
    db_session.add(alias)
    await db_session.commit()
    fetched = await db_session.get(ToolAlias, alias.id)
    assert fetched is not None
    assert fetched.raw_name == "payments_v1"
    assert fetched.canonical_name == "payments"


async def test_round_trip_fingerprint_baseline(db_session, seeded_agent) -> None:
    fp = FingerprintBaseline(
        id=uuid.uuid4(),
        agent_id=seeded_agent.id,
        baseline_counts={"a": 80, "b": 20},
        observed_counts={"a": 70, "b": 30},
        n_observed=100,
        chi_square_threshold=13.82,
    )
    db_session.add(fp)
    await db_session.commit()
    fetched = await db_session.get(FingerprintBaseline, fp.id)
    assert fetched is not None
    assert fetched.baseline_counts == {"a": 80, "b": 20}
    assert fetched.observed_counts == {"a": 70, "b": 30}

