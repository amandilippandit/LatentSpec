"""Violation table — the audit log of regression detections (§8.1).

Each row links an offending trace to the invariant it violated, with a
JSONB `details` payload carrying the §4.3 root-cause hypothesis, segment
breakdown, and side-by-side comparison data.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Index, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from latentspec.db import Base
from latentspec.models.invariant import Severity

if TYPE_CHECKING:
    from latentspec.models.invariant import Invariant


class Violation(Base):
    __tablename__ = "violations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    invariant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("invariants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    trace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("traces.id", ondelete="CASCADE"), nullable=False
    )

    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    severity: Mapped[Severity] = mapped_column(Enum(Severity, name="severity"), nullable=False)
    acknowledged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    invariant: Mapped["Invariant"] = relationship(back_populates="violations")

    __table_args__ = (
        Index("ix_violations_invariant_detected", "invariant_id", "detected_at"),
    )

    def __repr__(self) -> str:
        return f"<Violation id={self.id} inv={self.invariant_id} severity={self.severity.value}>"
