"""AutoGPT-style trace simulator.

Architecture (AutoGPT v0.5 documented loop):
  parse_goal → loop[ think → act(N) → observe → memory_write ]
             → finalize

Action choices come from a 20+ tool repertoire, sampled per step. Loop
depth varies wildly (3 - 15 iterations). Failed actions branch into
recovery sub-loops. This is the **hardest** case for the miner because
the trace shape is highly variable trace-to-trace.

Expected recoveries:
  - composition: parse_goal must precede every observe
  - statistical: think_step latency stable
  - negative: never `delete_workspace` (we don't put it in the repertoire)
"""

from __future__ import annotations

import random
import uuid
from datetime import UTC, datetime, timedelta

from latentspec.schemas.trace import (
    AgentResponseStep,
    AgentThoughtStep,
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)


ACTIONS = [
    "web_search", "fetch_url", "read_file", "write_file",
    "run_python", "shell_command", "summarize",
    "send_email", "schedule_event", "browse",
    "ask_user", "estimate_cost", "split_task",
    "merge_results", "validate_output", "search_memory",
    "memory_write", "evaluate_progress", "ask_human", "wait",
]


GOALS = [
    "find recent news about LLM evaluation tools",
    "build a Python script that downloads weather data",
    "compare prices of three CRM products",
    "draft a blog post about AI agents",
    "research best practices for prompt evaluation",
]


def _latency(mean: float) -> int:
    return max(10, int(random.gauss(mean, mean * 0.4)))


def _build(idx: int) -> NormalizedTrace:
    goal = random.choice(GOALS)
    started_at = datetime.now(UTC) - timedelta(minutes=random.randint(0, 60 * 24))
    steps = [UserInputStep(content=goal)]

    steps.append(
        ToolCallStep(
            tool="parse_goal",
            args={"goal": goal},
            latency_ms=_latency(180),
            result_status="success",
        )
    )

    iterations = random.randint(3, 15)
    for i in range(iterations):
        steps.append(AgentThoughtStep(content=f"Iteration {i}: planning next action..."))
        n_actions = random.randint(1, 4)
        for _ in range(n_actions):
            tool = random.choice(ACTIONS)
            failed = random.random() < 0.05
            steps.append(
                ToolCallStep(
                    tool=tool,
                    args={"iter": i, "fragment": random.randint(0, 99)},
                    latency_ms=_latency(450),
                    result_status="error" if failed else "success",
                )
            )
            if failed:
                # Recovery sub-loop: ~30% of failures trigger a retry chain
                if random.random() < 0.3:
                    steps.append(
                        ToolCallStep(
                            tool=tool,
                            args={"iter": i, "retry": True},
                            latency_ms=_latency(700),
                            result_status="success" if random.random() > 0.05 else "error",
                        )
                    )
        steps.append(
            ToolCallStep(
                tool="observe",
                args={"iter": i},
                latency_ms=_latency(95),
                result_status="success",
            )
        )
        steps.append(
            ToolCallStep(
                tool="memory_write",
                args={"persist": True, "iter": i},
                latency_ms=_latency(110),
                result_status="success",
            )
        )

    steps.append(
        ToolCallStep(
            tool="finalize",
            args={"goal": goal[:40]},
            latency_ms=_latency(160),
            result_status="success",
        )
    )
    steps.append(AgentResponseStep(content=f"Goal complete: {goal[:60]}..."))

    duration = sum(getattr(s, "latency_ms", 0) or 0 for s in steps)
    return NormalizedTrace(
        trace_id=f"autogpt-{idx:05d}-{uuid.uuid4().hex[:6]}",
        agent_id="autogpt-style-agent",
        timestamp=started_at,
        ended_at=started_at + timedelta(milliseconds=duration),
        steps=steps,
        metadata=TraceMetadata(model="gpt-style", version="autogpt-v0.5"),
    )


def generate(n_traces: int, *, seed: int = 23) -> list[NormalizedTrace]:
    random.seed(seed)
    return [_build(i) for i in range(n_traces)]
