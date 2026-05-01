"""§3.2 invariant schema and §3.4 candidate types.

`MinedInvariant` is the rich object emitted by Stage 3 formalization.
`InvariantCandidate` is the lighter intermediate produced by each mining
track before cross-validation. `InvariantOut` is the API response shape.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from latentspec.models.invariant import InvariantStatus, InvariantType, Severity


class InvariantCandidate(BaseModel):
    """A pattern discovered by one mining track, pre-cross-validation."""

    model_config = ConfigDict(use_enum_values=False)

    type: InvariantType
    description: str
    formal_rule: str
    evidence_trace_ids: list[str] = Field(default_factory=list)
    support: float = Field(ge=0.0, le=1.0, description="fraction of traces exhibiting the rule")
    consistency: float = Field(
        ge=0.0, le=1.0, description="1 - (violation_rate within training set)"
    )
    severity: Severity = Severity.MEDIUM
    discovered_by: Literal["statistical", "llm", "both"]
    extra: dict[str, Any] = Field(default_factory=dict)


class MinedInvariant(BaseModel):
    """Stage 3 output — the formalized invariant ready to write to DB.

    Mirrors the §3.2 example object exactly, plus the §3.4 confidence
    breakdown so downstream UI can show *why* a confidence score is what it is.
    """

    model_config = ConfigDict(use_enum_values=False)

    invariant_id: str
    type: InvariantType
    description: str
    formal_rule: str
    confidence: float = Field(ge=0.0, le=1.0)
    support_score: float = Field(ge=0.0, le=1.0)
    consistency_score: float = Field(ge=0.0, le=1.0)
    cross_val_bonus: float = Field(ge=0.0, le=0.2)
    clarity_score: float = Field(ge=0.0, le=1.0)
    evidence_count: int = Field(ge=0)
    violation_count: int = Field(ge=0)
    discovered_at: datetime
    status: InvariantStatus
    severity: Severity
    discovered_by: Literal["statistical", "llm", "both", "pack"]
    evidence_trace_ids: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class InvariantOut(BaseModel):
    """API response shape for an invariant row."""

    model_config = ConfigDict(from_attributes=True, use_enum_values=False)

    id: UUID
    agent_id: UUID
    type: InvariantType
    description: str
    formal_rule: str
    confidence: float
    severity: Severity
    status: InvariantStatus
    evidence_count: int
    violation_count: int
    violation_rate: float
    discovered_at: datetime
    last_checked_at: datetime | None
    discovered_by: str
    params: dict[str, Any] = Field(default_factory=dict)
