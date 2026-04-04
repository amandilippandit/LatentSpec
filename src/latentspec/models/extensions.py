"""Persistent state for the seven coverage extensions + production hardening.

Eight tables:

  - tool_aliases          maps raw tool names to canonical IDs per agent
  - agent_versions        registry of agent versions for multi-version handling
  - fingerprint_baselines per-agent fingerprint distribution snapshots
  - cluster_centroids     per-agent k-means centroids + frozen vectorizer state
  - drift_state           per-(agent, invariant) Page-Hinkley + CUSUM state
  - sessions              multi-turn session storage
  - synthetic_review_queue active-learning HITL queue (persisted)
  - mining_jobs           background mining job state + progress
  - calibration_results   per-agent learned thresholds replacing defaults

These match the gaps the previous build pass left in process memory only.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from latentspec.db import Base


# ---- enums ---------------------------------------------------------------


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobKind(str, enum.Enum):
    MINING = "mining"
    CLUSTER_MINING = "cluster_mining"
    SESSION_MINING = "session_mining"
    CALIBRATION = "calibration"
    SYNTHETIC_GENERATION = "synthetic_generation"
    PACK_AUTOFIT = "pack_autofit"


class ReviewDecision(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"


# ---- tables --------------------------------------------------------------


class ToolAlias(Base):
    """`(agent_id, raw_tool_name) -> canonical_tool_name`.

    Populated by the canonicaliser and consulted by the trace ingest pass
    so `payments_v1` and `payments-v1` collapse to one canonical name.
    """

    __tablename__ = "tool_aliases"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    raw_name: Mapped[str] = mapped_column(String(256), nullable=False)
    canonical_name: Mapped[str] = mapped_column(String(256), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    method: Mapped[str] = mapped_column(String(32), nullable=False, default="exact")
    name_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(384), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("agent_id", "raw_name", name="uq_tool_aliases_agent_raw"),
        Index("ix_tool_aliases_canonical", "agent_id", "canonical_name"),
    )


class AgentVersion(Base):
    """Per-agent version registry — tracks tool-set changes across deploys."""

    __tablename__ = "agent_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_tag: Mapped[str] = mapped_column(String(64), nullable=False)
    tool_repertoire: Mapped[list[str]] = mapped_column(
        ARRAY(String), nullable=False, default=list
    )
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    parent_version_tag: Mapped[str | None] = mapped_column(String(64), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("agent_id", "version_tag", name="uq_agent_versions_agent_tag"),
    )


class FingerprintBaseline(Base):
    """Snapshot of an agent's fingerprint distribution + observed-window counts.

    A row is created the first time mining runs and updated as the
    streaming detector observes new traces.
    """

    __tablename__ = "fingerprint_baselines"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    baseline_counts: Mapped[dict[str, int]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    observed_counts: Mapped[dict[str, int]] = mapped_column(
        JSONB, nullable=False, default=dict
    )
    n_observed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chi_square_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=13.82)
    last_drift_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_drift_chi: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class ClusterCentroid(Base):
    """Persisted k-means centroid for one (agent, cluster_id) pair.

    The vectorizer's frozen vocabulary is stored alongside so traces
    arriving after the mining run can be vectorised consistently.
    """

    __tablename__ = "cluster_centroids"

