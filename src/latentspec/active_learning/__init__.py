"""Active-learning loop — bootstrap mining when real traces are scarce.

Two surfaces:

  - **Synthetic trace generator** — given an agent description (its tools,
    its purpose, sample inputs), Claude is prompted to emit plausible §3.2
    traces. These are tagged `is_synthetic=True` and queued for human
    approval before they enter the real mining corpus.

  - **HITL review queue** — `ReviewQueue` holds candidate synthetic traces
    plus their generation provenance. A human approves / edits / rejects
    each one; approved traces flow into the next mining run with a
    `synthetic_provenance` audit field.

This bootstraps two cases that pure mining can't reach:

  1. Brand-new agents with no real traffic yet (the day-1 cold-start).
  2. Rare-but-critical paths the agent hasn't taken in production
     (e.g. emergency_escalate flows; mining never sees enough of them
     to be statistically meaningful).
"""

from latentspec.active_learning.queue import (
    ReviewDecision,
    ReviewQueue,
    SyntheticTrace,
)
from latentspec.active_learning.synthesis import (
    AgentSpec,
    SyntheticTraceGenerator,
    generate_synthetic_traces,
)

__all__ = [
    "AgentSpec",
    "ReviewDecision",
    "ReviewQueue",
    "SyntheticTrace",
    "SyntheticTraceGenerator",
    "generate_synthetic_traces",
]
