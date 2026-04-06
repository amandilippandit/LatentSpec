"""Normalizer for LangChain run payloads (§5.2 P0).

LangChain emits runs as nested chain/agent/tool/llm events. This module
flattens that hierarchy into the §3.2 ordered `steps[]` array. Designed to
accept either:
  - a single LangChain Run dict (with nested `child_runs`)
  - a flat list of run events (what the LangSmith callback emits)

The mapping is deliberately conservative — we only emit step types we can
ground in the §3.2 schema:
    chain/agent input  → user_input
    tool start/end     → tool_call (paired)
    llm/agent output   → agent_response
    intermediate think → agent_thought
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from latentspec.normalizers.base import Normalizer, NormalizerError, registry
from latentspec.schemas.trace import (
    AgentResponseStep,
    AgentThoughtStep,
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    TraceStep,
    UserInputStep,
)


def _to_iso(value: Any, fallback: datetime) -> datetime:
    if value is None:
        return fallback
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return fallback
    return fallback


def _latency_ms(start: datetime, end: datetime | None) -> int | None:
    if end is None:
        return None
    delta = (end - start).total_seconds() * 1000
    return max(0, int(delta))


def _walk_runs(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Depth-first traversal of a LangChain Run tree.

    Returns runs in start-time order so the resulting steps preserve causality.
    """
    flat: list[dict[str, Any]] = [run]
    for child in run.get("child_runs") or []:
        flat.extend(_walk_runs(child))
    flat.sort(key=lambda r: r.get("start_time") or "")
    return flat


def _extract_user_input(root: dict[str, Any]) -> str | None:
    inputs = root.get("inputs") or {}
    for key in ("input", "question", "query", "prompt", "messages"):
        if key in inputs and inputs[key]:
            value = inputs[key]
            if isinstance(value, list):
                # last user message in a chat history
                for msg in reversed(value):
                    if isinstance(msg, dict) and msg.get("role") in (None, "user", "human"):
                        return str(msg.get("content") or msg.get("text") or "")
            return str(value)
    return None


def _extract_agent_response(root: dict[str, Any]) -> str | None:
    outputs = root.get("outputs") or {}
    for key in ("output", "answer", "result", "response", "content"):
        if key in outputs and outputs[key]:
            value = outputs[key]
            if isinstance(value, dict):
                return str(value.get("content") or value.get("text") or value)
