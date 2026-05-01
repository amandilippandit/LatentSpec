"""Claude-driven synthetic trace generator.

Given an `AgentSpec` (tools the agent has, what it does, sample user
inputs) we ask Claude to emit N plausible §3.2 traces in the schema's
exact shape. The output goes through the same Pydantic validation as
production traces, so malformed candidates never reach mining.

These are NOT a replacement for real traces. They're a bootstrap. Real
production traffic always wins when it's available; synthetic traces
fill the gap for cold-start, rare-path, and adversarial-coverage cases.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import ValidationError

from latentspec.config import get_settings
from latentspec.schemas.trace import NormalizedTrace, TraceMetadata

log = logging.getLogger(__name__)


@dataclass
class AgentSpec:
    """Description of an agent's purpose, tools, and seed inputs.

    Used as the prompt input to the synthetic trace generator. The richer
    this spec, the better the synthetic traces — but even a minimal spec
    (name + tool list) produces reasonable bootstrap data.
    """

    name: str
    purpose: str
    tools: list[str]
    sample_user_inputs: list[str] = field(default_factory=list)
    user_segments: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    typical_session_length: tuple[int, int] = (3, 8)


SYSTEM_PROMPT = """\
You are LatentSpec's synthetic trace generator. Given an agent
specification, produce realistic §3.2-shaped traces of the agent
operating on plausible user inputs.

Return ONLY a JSON array of traces. Each trace MUST conform to:
  {
    "trace_id": "<unique id>",
    "agent_id": "<the agent's id>",
    "timestamp": "<ISO 8601 datetime>",
    "steps": [
      {"type": "user_input", "content": "..."},
      {"type": "tool_call", "tool": "<one of agent.tools>",
       "args": {...}, "latency_ms": <int>, "result_status": "success"|"error"},
      {"type": "agent_response", "content": "..."}
    ],
    "metadata": {"model": "...", "version": "...",
                 "user_segment": "<one of agent.user_segments>",
                 "locale": "..."}
  }

Rules:
1. Only invoke tools listed in `agent.tools`. Never invent tools.
2. Cover the diversity in `agent.sample_user_inputs` and
   `agent.user_segments`.
3. Include the implicit ordering / conditional behaviors a real agent
   would learn (auth before sensitive calls, segment-keyed routing,
   error-path divergences).
4. Generate `n_traces` traces, length range `agent.typical_session_length`.
5. Latencies between 30–800ms; ~1% of tool calls should be errors.
6. NEVER invoke a tool listed in `agent.forbidden_actions`.
"""


@dataclass
class SyntheticTraceGenerator:
    """Async generator backed by the Anthropic API."""

    model: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.4

    async def generate(
        self,
        spec: AgentSpec,
        *,
        n_traces: int = 20,
    ) -> list[NormalizedTrace]:
        settings = get_settings()
        if not settings.anthropic_api_key:
            log.warning("ANTHROPIC_API_KEY not set; cannot generate synthetic traces")
            return []

        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        user_message = json.dumps(
            {
                "agent": {
                    "name": spec.name,
                    "agent_id": spec.name.lower().replace(" ", "-"),
                    "purpose": spec.purpose,
                    "tools": spec.tools,
                    "sample_user_inputs": spec.sample_user_inputs,
                    "user_segments": spec.user_segments,
                    "forbidden_actions": spec.forbidden_actions,
                    "typical_session_length": list(spec.typical_session_length),
                },
                "n_traces": n_traces,
            },
            ensure_ascii=False,
        )
        try:
            resp = await client.messages.create(
                model=self.model or settings.anthropic_model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as e:  # noqa: BLE001
            log.warning("synthetic generation failed: %s", e)
            return []

        text = "".join(getattr(b, "text", "") or "" for b in resp.content).strip()
        return _parse_traces(text)


def _parse_traces(raw: str) -> list[NormalizedTrace]:
    if not raw:
        return []
    cleaned = raw
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1:
        # Sometimes Claude returns an object with `traces` key
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1:
            return []
        try:
            obj = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return []
        items = obj.get("traces") or []
    else:
        try:
            items = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return []

    out: list[NormalizedTrace] = []
    for item in items:
        try:
            out.append(NormalizedTrace.model_validate(item))
        except ValidationError as e:
            log.debug("dropping malformed synthetic trace: %s", e)
            continue
    return out


async def generate_synthetic_traces(
    spec: AgentSpec, *, n_traces: int = 20
) -> list[NormalizedTrace]:
    """Module-level convenience: one shot, returns parsed traces."""
    return await SyntheticTraceGenerator().generate(spec, n_traces=n_traces)


# ---- Deterministic generator for tests / no-API-key environments --------


def deterministic_synthetic_traces(
    spec: AgentSpec, *, n_traces: int = 20, seed: int = 13
) -> list[NormalizedTrace]:
    """Pure-Python fallback that emits structurally plausible traces.

    No LLM calls. Useful for unit tests and air-gapped deployments where
    Anthropic API access isn't available.
    """
    import random

    from latentspec.schemas.trace import (
        AgentResponseStep,
        ToolCallStep,
        UserInputStep,
    )

    rng = random.Random(seed)
    traces: list[NormalizedTrace] = []
    inputs = spec.sample_user_inputs or [f"Help me with {spec.purpose}"]
    segments = spec.user_segments or ["default"]

    for i in range(n_traces):
        n_steps = rng.randint(*spec.typical_session_length)
        user_input = rng.choice(inputs)
        segment = rng.choice(segments)
        steps: list[Any] = [UserInputStep(content=user_input)]
        # Synthesize a plausible chain by sampling tools sequentially.
        used_tools = []
        for _ in range(min(n_steps - 2, len(spec.tools))):
            tool = rng.choice(spec.tools)
            steps.append(
                ToolCallStep(
                    tool=tool,
                    args={"step": len(used_tools)},
                    latency_ms=rng.randint(30, 800),
                    result_status="success" if rng.random() > 0.01 else "error",
                )
            )
            used_tools.append(tool)
        steps.append(AgentResponseStep(content=f"Done with {spec.purpose}."))

        traces.append(
            NormalizedTrace(
                trace_id=f"synth-{uuid.uuid4().hex[:10]}",
                agent_id=spec.name.lower().replace(" ", "-"),
                timestamp=datetime.now(UTC) - timedelta(minutes=rng.randint(0, 5000)),
                ended_at=datetime.now(UTC),
                steps=steps,
                metadata=TraceMetadata(
                    user_segment=segment, version="synthetic-v1"
                ),
            )
        )
    return traces
