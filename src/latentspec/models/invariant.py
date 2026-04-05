"""Invariant table — the core product output (§8.1).

Each row implements the §3.2 invariant schema: type, description, formal_rule,
confidence, severity, evidence_trace_ids, lifecycle metadata. The eight types
come from §3.3 taxonomy. The status enum maps to §3.4 three-band gating.

`description_embedding` (pgvector) backs §8.2 invariant semantic search.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from latentspec.db import Base

if TYPE_CHECKING:
    from latentspec.models.agent import Agent
    from latentspec.models.violation import Violation


class InvariantType(str, enum.Enum):
    """The eight invariant types from §3.3 taxonomy."""

    ORDERING = "ordering"
    CONDITIONAL = "conditional"
    NEGATIVE = "negative"
    STATISTICAL = "statistical"
    OUTPUT_FORMAT = "output_format"
    TOOL_SELECTION = "tool_selection"
    STATE = "state"
    COMPOSITION = "composition"


class Severity(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InvariantStatus(str, enum.Enum):
    """Three-band gating from §3.4."""

    PENDING = "pending"
    ACTIVE = "active"
    REJECTED = "rejected"


# Embedding dimension matches the model used for description vectorization.
# Default: text-embedding-3-small (1536) — picked here as a stable default.
INVARIANT_EMBEDDING_DIM = 1536


class Invariant(Base):
    __tablename__ = "invariants"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True
    )

    type: Mapped[InvariantType] = mapped_column(
        Enum(InvariantType, name="invariant_type"), nullable=False, index=True
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    formal_rule: Mapped[str] = mapped_column(Text, nullable=False)

    # Confidence breakdown (§3.4): final score plus the four sub-scores
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    support_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    consistency_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cross_val_bonus: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    clarity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    severity: Mapped[Severity] = mapped_column(
        Enum(Severity, name="severity"), nullable=False, default=Severity.MEDIUM
    )
    status: Mapped[InvariantStatus] = mapped_column(
        Enum(InvariantStatus, name="invariant_status"),
        nullable=False,
        default=InvariantStatus.PENDING,
        index=True,
    )

    evidence_trace_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )
    evidence_count: Mapped[int] = mapped_column(default=0)
    violation_count: Mapped[int] = mapped_column(default=0)
    violation_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_checked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # pgvector for semantic search across invariants (§8.2)
    description_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(INVARIANT_EMBEDDING_DIM), nullable=True
    )

    # Sourcing metadata: which tracks (statistical / llm / both) found this candidate
    discovered_by: Mapped[str] = mapped_column(String(32), nullable=False, default="both")

    # Structured rule parameters (read by the runtime checker). For ordering
    # invariants this is `{"tool_a": "...", "tool_b": "..."}`; for statistical
    # latency rules it's `{"tool": "...", "threshold": 500, "percentile": 99}`,
    # etc. Lets us check rules without parsing `formal_rule` strings.
    params: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    agent: Mapped["Agent"] = relationship(back_populates="invariants")
    violations: Mapped[list["Violation"]] = relationship(
        back_populates="invariant", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_invariants_agent_status", "agent_id", "status"),
        Index("ix_invariants_agent_type", "agent_id", "type"),
    )

    def __repr__(self) -> str:
        return (
            f"<Invariant id={self.id} type={self.type.value} "
            f"conf={self.confidence:.2f} status={self.status.value}>"
        )
