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


def current_context() -> GuardrailContext | None:
    stack: list[GuardrailContext] = getattr(_thread_local, "stack", [])
    return stack[-1] if stack else None


# ----- @guardrail decorator ----------------------------------------------


def _check_against(
    rules: RuleSet, trace: NormalizedTrace, *, max_check_ms: int
) -> tuple[list, bool]:
    """Run every applicable rule under a strict latency budget.

    Returns (failures, fail_open). Each failure is a `CheckResult` with a
    FAIL outcome.
    """
    deadline = time.perf_counter() + (max_check_ms / 1000.0)
    failures = []
    fail_open = False
    for inv in rules.invariants:
        if time.perf_counter() >= deadline:
            fail_open = True
            break
        try:
            result = dispatch(inv, trace)
        except Exception as e:  # noqa: BLE001
            log.debug("guardrail dispatch error on %s: %s", inv.id, e)
            continue
        if result.outcome in (CheckOutcome.FAIL, CheckOutcome.WARN):
            failures.append(result)
    return failures, fail_open


def guardrail(
    rules: RuleSet,
    *,
    fail_on: _FailOnLevel = "critical",
    max_check_ms: int = 80,
    name: str | None = None,
) -> Callable[..., Any]:
    """Decorator that enforces `rules` on the wrapped tool function.

    Pre-call:  build a candidate trace from collected context + a synthetic
               trailing tool_call (the about-to-run call). Run dispatch.
               Raise GuardrailViolation when policy demands.
    Post-call: append the actual ToolCallStep with measured latency/status.
    """

    def _decorate(target: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or target.__name__

        if inspect.iscoroutinefunction(target):

            @functools.wraps(target)
            async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
                _enforce_pre(rules, tool_name, args, kwargs, fail_on, max_check_ms, target)
                start = time.perf_counter()
                status = "success"
                try:
                    return await target(*args, **kwargs)
                except BaseException:
                    status = "error"
                    raise
                finally:
                    _record_post(tool_name, args, kwargs, start, status, target)

            return _async_wrapper

        @functools.wraps(target)
        def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            _enforce_pre(rules, tool_name, args, kwargs, fail_on, max_check_ms, target)
            start = time.perf_counter()
            status = "success"
            try:
                return target(*args, **kwargs)
            except BaseException:
                status = "error"
                raise
            finally:
                _record_post(tool_name, args, kwargs, start, status, target)

        return _sync_wrapper

    return _decorate


def _args_to_dict(target: Callable[..., Any], args: tuple, kwargs: dict) -> dict[str, Any]:
    try:
        sig = inspect.signature(target)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return {k: _safe(v) for k, v in bound.arguments.items()}
    except (TypeError, ValueError):
        return {f"arg_{i}": _safe(v) for i, v in enumerate(args)} | {
            k: _safe(v) for k, v in kwargs.items()
        }


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 1000 else value[:1000] + "…"
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in list(value)[:25]]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in list(value.items())[:25]}
    return str(value)[:1000]


def _enforce_pre(
    rules: RuleSet,
    tool_name: str,
    args: tuple,
    kwargs: dict,
    fail_on: _FailOnLevel,
    max_check_ms: int,
    target: Callable[..., Any],
) -> None:
    ctx = current_context()
    candidate_steps: list[TraceStep]
    if ctx is None:
        candidate_steps = [
            ToolCallStep(tool=tool_name, args=_args_to_dict(target, args, kwargs))
        ]
        candidate_trace = NormalizedTrace(
            trace_id="guardrail-precall",
            agent_id=rules.agent_id,
            timestamp=datetime.now(UTC),
            steps=candidate_steps,
            metadata=TraceMetadata(),
        )
    else:
        candidate_steps = list(ctx.steps) + [
            ToolCallStep(tool=tool_name, args=_args_to_dict(target, args, kwargs))
        ]
        candidate_trace = NormalizedTrace(
            trace_id="guardrail-precall",
            agent_id=ctx.agent_id,
            timestamp=ctx.started_at,
            steps=candidate_steps,
            metadata=ctx.metadata,
        )

    failures, fail_open = _check_against(rules, candidate_trace, max_check_ms=max_check_ms)
    if fail_open:
        log.warning(
            "guardrail check exceeded %dms budget — %d rules pending; failing open",
            max_check_ms,
            len(rules) - len(failures),
        )

    for f in failures:
        if f.outcome == CheckOutcome.WARN and fail_on != "warn":
            log.info("guardrail warn: %s — %s", f.invariant_description, f.details)
            continue
        if not _should_raise(f.severity, fail_on):
            log.info("guardrail violation (below fail_on): %s", f.invariant_description)
            continue
        observed = (f.details.observed if f.details else "rule violated")
        raise GuardrailViolation(
            f"GuardrailViolation: {f.invariant_description}",
            invariant_id=f.invariant_id,
            invariant_description=f.invariant_description,
            severity=f.severity,
            observed=observed,
        )


def _record_post(
    tool_name: str,
    args: tuple,
    kwargs: dict,
    start: float,
    status: str,
    target: Callable[..., Any],
) -> None:
    ctx = current_context()
    if ctx is None:
        return
    latency_ms = max(0, int((time.perf_counter() - start) * 1000))
    ctx.steps.append(
        ToolCallStep(
            tool=tool_name,
            args=_args_to_dict(target, args, kwargs),
            latency_ms=latency_ms,
            result_status=status,
        )
    )


# ----- Convenience context manager wrapping a guarded turn ----------------


class guarded_turn:  # noqa: N801 — public API casing
    """Context manager that opens a guardrail context for one agent turn.

        rules = RuleSet.from_local_file("booking-agent", "invariants.json")
        with latentspec.guarded_turn(rules, user_input="book a flight"):
            search_flights(...)   # @guardrail-decorated
            book_flight(...)
    """

    def __init__(
        self,
        rules: RuleSet,
        *,
        user_input: str | None = None,
        version: str | None = None,
        user_segment: str | None = None,
    ) -> None:
        self._ctx = GuardrailContext(
            agent_id=rules.agent_id,
            user_input=user_input,
            metadata=TraceMetadata(version=version, user_segment=user_segment),
        )
        if user_input is not None:
            self._ctx.steps.append(UserInputStep(content=user_input))

    def __enter__(self) -> "guarded_turn":
        push_context(self._ctx)
        return self

    def add_response(self, content: str) -> None:
        self._ctx.steps.append(AgentResponseStep(content=content))

    def __exit__(self, exc_type, exc, tb) -> None:
        pop_context()
        if exc is not None:
            return None
        # On clean exit, ship the assembled trace via the SDK if it's initialised.
        from latentspec.sdk.client import get_client

        client = get_client()
        if client is not None and self._ctx.steps:
            client.record(self._ctx.to_trace())
        return None
