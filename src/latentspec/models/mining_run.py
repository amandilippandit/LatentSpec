"""MiningRun table — operations log for each Stage 2 pipeline invocation (§8.1).

Tracks parameters (in `config`) and results (`traces_analyzed`,
`invariants_discovered`) for every batch mining job. Without this, debugging
sudden discovery-rate drops requires log archaeology.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from latentspec.db import Base

if TYPE_CHECKING:
    from latentspec.models.agent import Agent


class MiningRunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class MiningRun(Base):
    __tablename__ = "mining_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    traces_analyzed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    invariants_discovered: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    status: Mapped[MiningRunStatus] = mapped_column(
        Enum(MiningRunStatus, name="mining_run_status"),
        nullable=False,
        default=MiningRunStatus.PENDING,
    )
    error: Mapped[str | None] = mapped_column(nullable=True)

    agent: Mapped["Agent"] = relationship(back_populates="mining_runs")

    def __repr__(self) -> str:
        return (
            f"<MiningRun id={self.id} agent={self.agent_id} "
            f"status={self.status.value} discovered={self.invariants_discovered}>"
        )
