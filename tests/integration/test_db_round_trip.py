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


async def test_round_trip_cluster_centroid(db_session, seeded_agent) -> None:
    cc = ClusterCentroid(
        id=uuid.uuid4(),
        agent_id=seeded_agent.id,
        cluster_id=0,
        centroid=[0.1, 0.2, 0.3],
        n_traces=42,
        silhouette=0.61,
        vectorizer_state={"tool_idx": {"a": 0, "b": 1}},
    )
    db_session.add(cc)
    await db_session.commit()
    fetched = await db_session.get(ClusterCentroid, cc.id)
    assert fetched is not None
    assert fetched.cluster_id == 0
    assert fetched.centroid == [0.1, 0.2, 0.3]


async def test_round_trip_session(db_session, seeded_agent) -> None:
    sess = Session(
        id=uuid.uuid4(),
        agent_id=seeded_agent.id,
        external_id="ext-123",
        user_id="u-1",
        started_at=datetime.now(UTC),
        metadata_={"source": "integration-test"},
        n_turns=3,
    )
    db_session.add(sess)
    await db_session.commit()
    fetched = await db_session.get(Session, sess.id)
    assert fetched is not None
    assert fetched.external_id == "ext-123"
    assert fetched.metadata_ == {"source": "integration-test"}


async def test_round_trip_mining_job(db_session, seeded_agent) -> None:
    job = MiningJob(
        id=uuid.uuid4(),
        agent_id=seeded_agent.id,
        kind=JobKind.MINING,
        status=JobStatus.PENDING,
        config={"limit": 500},
    )
    db_session.add(job)
    await db_session.commit()
    fetched = await db_session.get(MiningJob, job.id)
    assert fetched is not None
    assert fetched.kind == JobKind.MINING
    assert fetched.status == JobStatus.PENDING
    assert fetched.config == {"limit": 500}


async def test_round_trip_calibration_result(db_session, seeded_agent) -> None:
    cal = CalibrationResult(
        id=uuid.uuid4(),
        agent_id=seeded_agent.id,
        n_traces_calibrated=120,
        mining_min_support=0.55,
        confidence_review_threshold=0.82,
        fingerprint_chi_square_threshold=11.34,
        drift_ph_threshold=9.5,
        distribution_summary={"n_distinct_tools": 14},
    )
    db_session.add(cal)
    await db_session.commit()
    fetched = await db_session.get(CalibrationResult, cal.id)
    assert fetched is not None
    assert fetched.mining_min_support == 0.55
    assert fetched.distribution_summary == {"n_distinct_tools": 14}


async def test_round_trip_synthetic_review_item(db_session, seeded_agent) -> None:
    item = SyntheticReviewItem(
        id=uuid.uuid4(),
        agent_id=seeded_agent.id,
        spec_name="booking-agent",
        trace_data={"steps": []},
        decision=ReviewDecision.PENDING,
    )
    db_session.add(item)
    await db_session.commit()
    fetched = await db_session.get(SyntheticReviewItem, item.id)
    assert fetched is not None
    assert fetched.decision == ReviewDecision.PENDING


async def test_round_trip_agent_version(db_session, seeded_agent) -> None:
    av = AgentVersion(
        id=uuid.uuid4(),
        agent_id=seeded_agent.id,
        version_tag="v2.1",
        tool_repertoire=["search", "book", "cancel"],
    )
    db_session.add(av)
    await db_session.commit()
    fetched = await db_session.get(AgentVersion, av.id)
    assert fetched is not None
    assert fetched.tool_repertoire == ["search", "book", "cancel"]
