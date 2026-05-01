"""Trace table — high-volume time-series storage for normalized agent traces.

The §3.2 normalized JSON sits in `trace_data` (JSONB). Hot fields are
denormalized (started_at, ended_at, step_count, status) for fast querying
without JSON parsing. Promoted to a TimescaleDB hypertable on started_at.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from latentspec.db import Base


class TraceStatus(str, enum.Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    PARTIAL = "partial"


class Trace(Base):
    __tablename__ = "traces"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"), nullable=False
    )

    # Normalized §3.2 schema lives here
    trace_data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    version_tag: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    session_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    cluster_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    step_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[TraceStatus] = mapped_column(
        Enum(TraceStatus, name="trace_status"), nullable=False, default=TraceStatus.SUCCESS
    )

    agent: Mapped["Agent"] = relationship(back_populates="traces")  # type: ignore[name-defined] # noqa: F821

    __table_args__ = (
        Index("ix_traces_agent_started", "agent_id", "started_at"),
        Index("ix_traces_version", "agent_id", "version_tag"),
    )

    def __repr__(self) -> str:
        return f"<Trace id={self.id} agent={self.agent_id} steps={self.step_count}>"
