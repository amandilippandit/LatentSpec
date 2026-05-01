"""Agent table — tenant-scoped registration of an AI agent (§8.1).

`org_id` is the multi-tenant boundary. Even though auth wires up in week 3,
the column ships in week 1 to avoid a painful migration later.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from latentspec.db import Base


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    framework: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    traces: Mapped[list["Trace"]] = relationship(  # type: ignore[name-defined] # noqa: F821
        back_populates="agent", cascade="all, delete-orphan"
    )
    invariants: Mapped[list["Invariant"]] = relationship(  # type: ignore[name-defined] # noqa: F821
        back_populates="agent", cascade="all, delete-orphan"
    )
    mining_runs: Mapped[list["MiningRun"]] = relationship(  # type: ignore[name-defined] # noqa: F821
        back_populates="agent", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Agent id={self.id} name={self.name!r} framework={self.framework!r}>"
