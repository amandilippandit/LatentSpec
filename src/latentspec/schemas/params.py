"""Per-InvariantType params schemas (§3.3).

Every invariant carries a `params` JSONB blob that the runtime checker reads.
Without schemas, the LLM track can emit malformed shapes that pass mining
and fail silently at check time. This module defines a Pydantic schema per
type and a `validate_params(type, params)` entry point that the formalizer,
the checker contract, and the LLM-output parser all call.

A type without a schema is passed through unchanged (so `output_format`,
which has no machine-checkable params, stays free-form).
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from latentspec.models.invariant import InvariantType


_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_./-]{0,127}$")
_KEYWORD_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


class ParamsValidationError(ValueError):
    """Raised when invariant params don't match the type's schema."""


def _check_tool(value: str, *, field: str) -> str:
    if not _TOOL_NAME_RE.match(value):
        raise ValueError(f"{field}: invalid tool name {value!r}")
    return value


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class OrderingParams(_Base):
    """A precedes B (length-2) or full chain ordering (length 3+)."""

    tool_a: str
    tool_b: str
    chain: list[str] | None = None
    co_occurrence: int | None = Field(default=None, ge=0)
    directionality: float | None = Field(default=None, ge=0.0, le=1.0)
    gap_consistency: float | None = Field(default=None, ge=0.0, le=1.0)
    pattern_length: int | None = Field(default=None, ge=2, le=8)

    @field_validator("tool_a", "tool_b")
    @classmethod
    def _validate_tool(cls, v: str) -> str:
        return _check_tool(v, field="tool_a/tool_b")

    @field_validator("chain")
    @classmethod
    def _validate_chain(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        if len(v) < 2:
            raise ValueError("chain must have at least 2 tools")
        for t in v:
            _check_tool(t, field="chain")
        return v


class ConditionalParams(_Base):
    """If user input contains keyword, the agent must invoke tool."""

    keyword: str
    tool: str
    keyword_traces: int | None = Field(default=None, ge=0)
    lift: float | None = None

    @field_validator("keyword")
    @classmethod
    def _validate_keyword(cls, v: str) -> str:
        v = v.lower()
        if not _KEYWORD_RE.match(v):
            raise ValueError(f"invalid keyword {v!r} — must be lowercase alphanumeric+underscore, length 2–64")
        return v

    @field_validator("tool")
    @classmethod
    def _validate_tool(cls, v: str) -> str:
        return _check_tool(v, field="tool")


class NegativeParams(_Base):
    """Agent never invokes a tool matching any of `forbidden_patterns`,
    OR closed-world: agent only uses tools in `allowed_repertoire`."""

    forbidden_patterns: list[str] | None = None
    allowed_repertoire: list[str] | None = None
    category: str | None = None
    closed_world: bool = False

    @field_validator("forbidden_patterns", "allowed_repertoire")
    @classmethod
    def _validate_tool_list(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        if not v:
            raise ValueError("tool list cannot be empty")
        for t in v:
            _check_tool(t, field="negative.tool_list")
        return v

    @model_validator(mode="after")
    def _exactly_one(self) -> "NegativeParams":
        if (self.forbidden_patterns is None) == (self.allowed_repertoire is None):
            raise ValueError(
                "negative params require exactly one of "
                "`forbidden_patterns` or `allowed_repertoire`"
            )
        if self.allowed_repertoire is not None:
            object.__setattr__(self, "closed_world", True)
        return self


class StatisticalParams(_Base):
    """One of three metrics: latency, success rate, feature envelope."""

    metric: Literal["latency_ms", "success_rate", "feature_envelope"]
    tool: str | None = None
    feature: str | None = None
    threshold: float | None = None
    rate: float | None = Field(default=None, ge=0.0, le=1.0)
    percentile: float | None = Field(default=None, ge=0.0, le=100.0)
    p1: float | None = None
    p99: float | None = None
    median: float | None = None
    samples: int | None = Field(default=None, ge=0)
    within_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("tool")
    @classmethod
    def _validate_tool(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _check_tool(v, field="tool")

    @model_validator(mode="after")
    def _required_for_metric(self) -> "StatisticalParams":
        if self.metric == "latency_ms":
            if not self.tool:
                raise ValueError("latency_ms requires `tool`")
            if self.threshold is None or self.threshold < 0:
                raise ValueError("latency_ms requires non-negative `threshold`")
        elif self.metric == "success_rate":
            if not self.tool:
                raise ValueError("success_rate requires `tool`")
            if self.rate is None:
                raise ValueError("success_rate requires `rate`")
        elif self.metric == "feature_envelope":
            if not self.feature:
                raise ValueError("feature_envelope requires `feature`")
            if self.p1 is None or self.p99 is None or self.p99 < self.p1:
                raise ValueError("feature_envelope requires `p1 <= p99`")
        return self


class StateParams(_Base):
    """After `terminator_tool`, the agent must not invoke any of `forbidden_after`."""

    terminator_tool: str
    forbidden_after: list[str] = Field(min_length=1)

    @field_validator("terminator_tool")
    @classmethod
    def _validate_terminator(cls, v: str) -> str:
        return _check_tool(v, field="terminator_tool")

    @field_validator("forbidden_after")
    @classmethod
    def _validate_after(cls, v: list[str]) -> list[str]:
        for t in v:
            _check_tool(t, field="forbidden_after")
        return v


class CompositionParams(_Base):
    """Upstream tool must precede every downstream invocation."""

    upstream_tool: str
    downstream_tool: str

    @field_validator("upstream_tool", "downstream_tool")
    @classmethod
    def _validate_tool(cls, v: str) -> str:
        return _check_tool(v, field="composition")


class ToolSelectionParams(_Base):
    """Segment-keyed routing: in `segment` use `expected_tool`, not `forbidden_tool`."""

    segment: str
    expected_tool: str
    forbidden_tool: str | None = None

    @field_validator("segment")
    @classmethod
    def _validate_segment(cls, v: str) -> str:
        if not _SEGMENT_RE.match(v):
            raise ValueError(f"invalid segment {v!r}")
        return v

    @field_validator("expected_tool", "forbidden_tool")
    @classmethod
    def _validate_tool(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return _check_tool(v, field="tool_selection")


class OutputFormatParams(_Base):
    """LLM-as-judge — opaque to the system, only the description matters."""

    rubric: str | None = Field(default=None, max_length=2048)


_SCHEMA_BY_TYPE: dict[InvariantType, type[BaseModel]] = {
    InvariantType.ORDERING: OrderingParams,
    InvariantType.CONDITIONAL: ConditionalParams,
    InvariantType.NEGATIVE: NegativeParams,
    InvariantType.STATISTICAL: StatisticalParams,
    InvariantType.STATE: StateParams,
    InvariantType.COMPOSITION: CompositionParams,
    InvariantType.TOOL_SELECTION: ToolSelectionParams,
    InvariantType.OUTPUT_FORMAT: OutputFormatParams,
}


def schema_for(invariant_type: InvariantType) -> type[BaseModel]:
    return _SCHEMA_BY_TYPE[invariant_type]


def validate_params(
    invariant_type: InvariantType, params: dict[str, Any]
) -> dict[str, Any]:
    """Validate and normalize params; raises ParamsValidationError on failure.

    Always run this before:
      - persisting an invariant to the DB
      - dispatching a checker against an invariant
      - merging two candidates in cross-validation
    """
    schema = _SCHEMA_BY_TYPE.get(invariant_type)
    if schema is None:
        return dict(params)
    try:
        validated = schema.model_validate(params)
    except Exception as e:  # ValidationError, ValueError, TypeError
        raise ParamsValidationError(
            f"params for {invariant_type.value} failed schema: {e}"
        ) from e
    return validated.model_dump(exclude_none=True)
