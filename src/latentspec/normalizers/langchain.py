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
            return str(value)
    return None


class LangChainNormalizer(Normalizer):
    """LangChain / LangGraph run → §3.2 normalized trace."""

    name = "langchain"

    # Run types that should emit a step
    TOOL_TYPES = {"tool"}
    LLM_TYPES = {"llm", "chat_model"}
    AGENT_TYPES = {"agent"}

    def normalize(self, payload: dict[str, Any], *, agent_id: str) -> NormalizedTrace:
        if not isinstance(payload, dict):
            raise NormalizerError("LangChain payload must be a dict")

        root_id = payload.get("id") or str(uuid.uuid4())
        now = datetime.now(UTC)
        started_at = _to_iso(payload.get("start_time"), now)
        ended_at = _to_iso(payload.get("end_time"), started_at)

        steps: list[TraceStep] = []

        # Step 1: user input from the root chain/agent run
        user_input = _extract_user_input(payload)
        if user_input is not None:
            steps.append(UserInputStep(content=user_input))

        # Step 2: tool calls and intermediate LLM thoughts (depth-first, time-ordered)
        for run in _walk_runs(payload):
            run_type = (run.get("run_type") or run.get("type") or "").lower()
            run_start = _to_iso(run.get("start_time"), started_at)
            run_end = _to_iso(run.get("end_time"), run_start)

            if run_type in self.TOOL_TYPES:
                tool_name = run.get("name") or run.get("serialized", {}).get("name") or "unknown"
                args = run.get("inputs") or {}
                error = run.get("error")
                steps.append(
                    ToolCallStep(
                        tool=str(tool_name),
                        args=args if isinstance(args, dict) else {"value": args},
                        latency_ms=_latency_ms(run_start, run_end),
                        result_status="error" if error else "success",
                        result=run.get("outputs"),
                    )
                )
            elif run_type in self.LLM_TYPES and run is not payload:
                # Intermediate LLM call — capture as agent_thought, not the final response.
                outputs = run.get("outputs") or {}
                gens = outputs.get("generations") if isinstance(outputs, dict) else None
                if gens:
                    text = ""
                    try:
                        text = str(gens[0][0].get("text", ""))
                    except (IndexError, AttributeError, KeyError):
                        text = ""
                    if text:
                        steps.append(AgentThoughtStep(content=text))

        # Step 3: final agent response
        agent_response = _extract_agent_response(payload)
        if agent_response is not None:
            steps.append(AgentResponseStep(content=agent_response))

        if not steps:
            raise NormalizerError("LangChain payload contained no extractable steps")

        metadata = TraceMetadata(
            model=payload.get("extra", {}).get("model")
            if isinstance(payload.get("extra"), dict)
            else None,
            version=payload.get("extra", {}).get("version")
            if isinstance(payload.get("extra"), dict)
            else None,
        )

        return NormalizedTrace(
            trace_id=str(root_id),
            agent_id=agent_id,
            timestamp=started_at,
            ended_at=ended_at,
            steps=steps,
            metadata=metadata,
        )


registry.register(LangChainNormalizer())
