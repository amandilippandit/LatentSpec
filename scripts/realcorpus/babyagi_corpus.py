"""BabyAGI-style trace simulator.

Architecture (from the public README):
  task_creation → task_prioritization → task_execution → result_storage
                                          ↓ (sometimes)
                                       reflect_on_result

Most runs follow the linear chain. ~15% diverge into reflection. Tools
all share a small vocabulary; expected discoveries:
  - ordering: prioritization always follows creation
  - ordering: execution always follows prioritization
  - ordering: result_storage always follows execution
  - statistical: execution latency stays moderate
"""

from __future__ import annotations

import random
import uuid
from datetime import UTC, datetime, timedelta

from latentspec.schemas.trace import (
    AgentResponseStep,
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)


GOAL_BANK = [
    "research GTM strategies for AI agents in India",
    "summarise the latest news on autonomous agents",
    "compile a list of open agent benchmarks",
    "draft a customer onboarding sequence",
    "outline a quarterly OKR document",
    "audit our internal pricing page",
]


def _latency(mean: float, *, sigma: float = 80.0, max_ms: int = 4000) -> int:
    return max(1, min(max_ms, int(random.gauss(mean, sigma))))


def _build(idx: int) -> NormalizedTrace:
    goal = random.choice(GOAL_BANK)
    started_at = datetime.now(UTC) - timedelta(minutes=random.randint(0, 60 * 24 * 14))
    steps = [UserInputStep(content=f"Objective: {goal}")]

    n_iterations = random.randint(1, 4)
    for i in range(n_iterations):
        steps.append(
            ToolCallStep(
                tool="task_creation",
                args={"objective": goal, "iter": i},
                latency_ms=_latency(220),
                result_status="success",
            )
        )
        steps.append(
            ToolCallStep(
                tool="task_prioritization",
                args={"queue_depth": random.randint(1, 8)},
                latency_ms=_latency(140),
                result_status="success",
            )
        )
        steps.append(
            ToolCallStep(
                tool="task_execution",
                args={"task": f"sub-task-{i}"},
                latency_ms=_latency(620),
                result_status="success" if random.random() > 0.04 else "error",
            )
        )
        steps.append(
            ToolCallStep(
                tool="result_storage",
                args={"persist": True},
                latency_ms=_latency(95),
                result_status="success",
            )
        )
        if random.random() < 0.15:
            steps.append(
                ToolCallStep(
                    tool="reflect_on_result",
                    args={"depth": "shallow"},
                    latency_ms=_latency(180),
                    result_status="success",
                )
            )

    steps.append(AgentResponseStep(content=f"Completed objective: {goal[:80]}..."))
    duration = sum(getattr(s, "latency_ms", 0) or 0 for s in steps)

    return NormalizedTrace(
        trace_id=f"babyagi-{idx:05d}-{uuid.uuid4().hex[:6]}",
        agent_id="babyagi-style-agent",
        timestamp=started_at,
        ended_at=started_at + timedelta(milliseconds=duration),
        steps=steps,
        metadata=TraceMetadata(model="claude-opus-4-7", version="babyagi-v1"),
    )


def generate(n_traces: int, *, seed: int = 17) -> list[NormalizedTrace]:
    random.seed(seed)
    return [_build(i) for i in range(n_traces)]
