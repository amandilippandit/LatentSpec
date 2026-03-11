"""OpenDevin-style trace simulator.

Architecture (OpenDevin paper, v0.10):
  spec_understanding → repo_exploration → edit_loop → run_tests → commit_or_repair

The edit loop iterates while tests fail. Tools span ~25 names across
filesystem, version control, execution, and search. Strong
keyword-conditional behaviour:
  - input mentions "tests fail" → repair_tests must be called
  - input mentions "performance" → benchmark must be called
  - input mentions "security" → audit_diff must be called

Expected recoveries:
  - ordering: read_file precedes every write_file
  - conditional: "test" → run_tests
  - conditional: "commit" → git_commit
  - composition: write_file → run_tests → git_commit
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


REQUESTS = [
    "Fix the failing tests in src/auth/",
    "Refactor the rate-limiter for performance — keep it under 200 lines",
    "Add a security audit for the JWT signing path",
    "Implement pagination for the orders endpoint",
    "The build is broken on CI — investigate and fix",
    "Migrate the sqlite layer to postgres",
    "Add an integration test for the OAuth callback",
]


def _latency(mean: float) -> int:
    return max(20, int(random.gauss(mean, mean * 0.3)))


def _build(idx: int) -> NormalizedTrace:
    request = random.choice(REQUESTS)
    rl = request.lower()
    is_test_request = "test" in rl
    is_perf_request = "performance" in rl
    is_security_request = "security" in rl
    is_migration_request = "migrat" in rl

    started_at = datetime.now(UTC) - timedelta(hours=random.randint(0, 240))
    steps = [UserInputStep(content=request)]

    steps.append(
        ToolCallStep(
            tool="parse_specification",
            args={"req": request},
            latency_ms=_latency(140),
            result_status="success",
        )
    )

    n_files_explored = random.randint(2, 8)
    for i in range(n_files_explored):
        steps.append(
            ToolCallStep(
                tool="read_file",
                args={"path": f"src/module_{i}.py"},
                latency_ms=_latency(60),
                result_status="success",
            )
