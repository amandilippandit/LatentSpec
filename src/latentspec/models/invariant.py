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
