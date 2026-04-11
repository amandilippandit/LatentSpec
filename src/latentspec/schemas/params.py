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
