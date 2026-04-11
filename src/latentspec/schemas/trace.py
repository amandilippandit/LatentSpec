"""§3.2 normalized trace schema.

The exact shape from the technical plan:

    {
      "trace_id": "abc-123",
      "agent_id": "booking-agent-v2",
      "timestamp": "2026-02-20T10:30:00Z",
      "steps": [
        {"type": "user_input", "content": "..."},
        {"type": "tool_call", "tool": "search_flights", "args": {...},
         "latency_ms": 342, "result_status": "success"},
        {"type": "agent_response", "content": "..."}
      ],
      "metadata": {"model": "claude-sonnet-4-5", "version": "v2.1"}
    }

Every framework integration in §5.2 converts *into* this schema.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Annotated, Any, Literal, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class StepType(str, enum.Enum):
    USER_INPUT = "user_input"
    TOOL_CALL = "tool_call"
    AGENT_RESPONSE = "agent_response"
    AGENT_THOUGHT = "agent_thought"
    SYSTEM = "system"


class _BaseStep(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")


class UserInputStep(_BaseStep):
    type: Literal[StepType.USER_INPUT] = StepType.USER_INPUT
    content: str


class ToolCallStep(_BaseStep):
    type: Literal[StepType.TOOL_CALL] = StepType.TOOL_CALL
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None
    result_status: str | None = None
    result: Any | None = None


class AgentResponseStep(_BaseStep):
    type: Literal[StepType.AGENT_RESPONSE] = StepType.AGENT_RESPONSE
    content: str


class AgentThoughtStep(_BaseStep):
    type: Literal[StepType.AGENT_THOUGHT] = StepType.AGENT_THOUGHT
    content: str


class SystemStep(_BaseStep):
    type: Literal[StepType.SYSTEM] = StepType.SYSTEM
    content: str


TraceStep = Annotated[
    Union[UserInputStep, ToolCallStep, AgentResponseStep, AgentThoughtStep, SystemStep],
    Field(discriminator="type"),
]


class TraceMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    version: str | None = None
    user_segment: str | None = None
    locale: str | None = None


class NormalizedTrace(BaseModel):
    """The §3.2 unified internal trace representation."""

    model_config = ConfigDict(populate_by_name=True)

    trace_id: str
    agent_id: str
    timestamp: datetime
    steps: list[TraceStep]
    metadata: TraceMetadata = Field(default_factory=TraceMetadata)
    ended_at: datetime | None = None


class TraceIn(BaseModel):
    """Wire format accepted by `POST /traces`.

    Accepts either:
      - a pre-normalized §3.2 trace (set `format="normalized"`)
      - a raw LangChain/OpenTelemetry/custom payload to be normalized server-side
    """

    model_config = ConfigDict(extra="forbid")

    agent_id: UUID
    format: Literal["normalized", "langchain", "raw_json"] = "normalized"
    payload: dict[str, Any]
    version_tag: str | None = None
