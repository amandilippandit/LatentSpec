"""Inline guardrail enforcement (production §4.1 hot path).

Use `@latentspec.guardrail` to *block* tool calls that would violate a
critical-severity invariant. Pass a `RuleSet` (loaded once at SDK init from
the API or from a local JSON file) and the decorator checks pre-conditions
before invoking the wrapped function.

    rules = latentspec.RuleSet.from_api(agent_id="booking-agent")

    @latentspec.guardrail(rules, fail_on="critical")
    def send_email(to: str, body: str) -> None:
        ...

    send_email("user@example.com", "hi")
    # raises GuardrailViolation if the active rule set says
    # `confirm_recipient` must be called before `send_email` and the
    # collector hasn't seen `confirm_recipient` yet in this trace.

Three failure modes:
  - `fail_on="critical"` — only critical-severity violations raise.
  - `fail_on="any"`      — any rule failure raises.
  - `fail_on="warn"`     — log only, never raise (observability mode).

The check uses the same dispatch layer as offline batch comparison —
identical semantics on PR-time and runtime. Latency budget is a hard
`max_check_ms` bound; checks that exceed it are skipped (fail-open).
"""

from __future__ import annotations

import functools
import inspect
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Iterable, Literal

from latentspec.checking.base import (
    CheckOutcome,
    InvariantSpec,
)
from latentspec.checking.dispatch import dispatch
from latentspec.models.invariant import InvariantType, Severity
from latentspec.schemas.trace import (
    AgentResponseStep,
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    TraceStep,
    UserInputStep,
)

log = logging.getLogger(__name__)

_FailOnLevel = Literal["critical", "high", "any", "warn"]


class GuardrailViolation(RuntimeError):
    """Raised when a guarded call would violate a critical invariant."""

    def __init__(
        self,
        message: str,
        *,
        invariant_id: str,
        invariant_description: str,
        severity: Severity,
        observed: str,
    ) -> None:
        super().__init__(message)
        self.invariant_id = invariant_id
        self.invariant_description = invariant_description
        self.severity = severity
        self.observed = observed


@dataclass
class RuleSet:
    """Bundle of `InvariantSpec` rules attached to an agent.

    Loaded once at SDK init and cached. Refresh via the SDK admin endpoints
    when mining produces a new active set; the cache invalidation pubsub from
    `streaming/cache.py` is the production refresh mechanism.
    """

    agent_id: str
    invariants: list[InvariantSpec] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __len__(self) -> int:
        return len(self.invariants)

    def filter(
        self,
        *,
        types: Iterable[InvariantType] | None = None,
        min_severity: Severity | None = None,
    ) -> list[InvariantSpec]:
        out = list(self.invariants)
        if types is not None:
            type_set = set(types)
            out = [i for i in out if i.type in type_set]
        if min_severity is not None:
            min_rank = _SEVERITY_RANK[min_severity]
            out = [i for i in out if _SEVERITY_RANK[i.severity] >= min_rank]
        return out

    @classmethod
    def from_invariants(cls, agent_id: str, invariants: Iterable[InvariantSpec]) -> "RuleSet":
        return cls(agent_id=agent_id, invariants=list(invariants))

    @classmethod
    def from_local_file(cls, agent_id: str, path: str) -> "RuleSet":
        import json
        from pathlib import Path

        raw = json.loads(Path(path).read_text())
        invs: list[InvariantSpec] = []
        for item in raw:
            invs.append(
                InvariantSpec(
                    id=str(item.get("invariant_id") or item.get("id") or ""),
                    type=InvariantType(item["type"]),
                    description=item["description"],
                    formal_rule=item.get("formal_rule") or "",
                    severity=Severity(item.get("severity", "medium")),
                    params=dict(item.get("params") or {}),
                )
            )
        return cls.from_invariants(agent_id, invs)

    @classmethod
    def from_api(
        cls,
        *,
        agent_id: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> "RuleSet":
        import os

        import httpx

        from latentspec.config import get_settings

        settings = get_settings()
        url = (api_base or os.environ.get("LATENTSPEC_API_BASE") or "").rstrip("/")
        if not url:
            raise RuntimeError("LATENTSPEC_API_BASE not configured")
        headers = {"Authorization": f"Bearer {api_key or os.environ.get('LATENTSPEC_API_KEY', '')}"}
        resp = httpx.get(
            f"{url}/invariants",
            params={"agent_id": agent_id, "status": "active"},
            headers=headers,
            timeout=10.0,
        )
        resp.raise_for_status()
        rows = resp.json()
        invs: list[InvariantSpec] = []
        for row in rows:
            invs.append(
                InvariantSpec(
                    id=str(row["id"]),
                    type=InvariantType(row["type"]),
                    description=row["description"],
                    formal_rule=row.get("formal_rule") or "",
                    severity=Severity(row.get("severity", "medium")),
                    params=dict(row.get("params") or {}),
                )
            )
        return cls.from_invariants(agent_id, invs)


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


def _should_raise(severity: Severity, fail_on: _FailOnLevel) -> bool:
    if fail_on == "warn":
        return False
    if fail_on == "any":
        return True
    if fail_on == "high":
        return _SEVERITY_RANK[severity] >= _SEVERITY_RANK[Severity.HIGH]
    # default critical
    return severity == Severity.CRITICAL


# ----- Guardrail context — collects steps so checkers can see history -----


@dataclass
class GuardrailContext:
    agent_id: str
    user_input: str | None = None
    steps: list[TraceStep] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: TraceMetadata = field(default_factory=TraceMetadata)

    def to_trace(self) -> NormalizedTrace:
        from uuid import uuid4

        return NormalizedTrace(
            trace_id=f"trace-{uuid4().hex[:12]}",
            agent_id=self.agent_id,
            timestamp=self.started_at,
            ended_at=datetime.now(UTC),
            steps=list(self.steps),
            metadata=self.metadata,
        )


_thread_local = threading.local()


def push_context(ctx: GuardrailContext) -> None:
    stack: list[GuardrailContext] = getattr(_thread_local, "stack", [])
    stack.append(ctx)
    _thread_local.stack = stack


def pop_context() -> GuardrailContext | None:
    stack: list[GuardrailContext] = getattr(_thread_local, "stack", [])
    if stack:
        ctx = stack.pop()
        _thread_local.stack = stack
        return ctx
    return None


