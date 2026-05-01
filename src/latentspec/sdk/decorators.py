"""@trace_tool and @trace decorators + step-collection context (§5.1).

Usage:

    @latentspec.trace_tool
    def search_flights(dest: str, date: str): ...

    # group several tool calls into one logical trace:
    with latentspec.trace(user_input="book a flight"):
        search_flights(...)
        book_flight(...)

The decorator captures `tool`, `args`, `latency_ms`, `result_status` for
every wrapped call and emits a §3.2 trace when the surrounding context
exits (or, if there's no surrounding `trace()` context, immediately).
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import logging
import time
import uuid
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from typing import Any, Callable

from latentspec.schemas.trace import (
    AgentResponseStep,
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    TraceStep,
    UserInputStep,
)

log = logging.getLogger(__name__)


class StepCollector:
    """Accumulates §3.2 steps for the duration of a `trace()` context."""

    __slots__ = ("trace_id", "agent_id", "started_at", "steps", "metadata")

    def __init__(
        self,
        *,
        trace_id: str | None = None,
        agent_id: str = "",
        metadata: TraceMetadata | None = None,
    ) -> None:
        self.trace_id = trace_id or f"trace-{uuid.uuid4().hex[:12]}"
        self.agent_id = agent_id
        self.started_at = datetime.now(UTC)
        self.steps: list[TraceStep] = []
        self.metadata = metadata or TraceMetadata()

    def add(self, step: TraceStep) -> None:
        self.steps.append(step)

    def to_trace(self) -> NormalizedTrace:
        return NormalizedTrace(
            trace_id=self.trace_id,
            agent_id=self.agent_id,
            timestamp=self.started_at,
            ended_at=datetime.now(UTC),
            steps=list(self.steps),
            metadata=self.metadata,
        )


_current: contextvars.ContextVar[StepCollector | None] = contextvars.ContextVar(
    "latentspec_collector", default=None
)


def current_collector() -> StepCollector | None:
    return _current.get()


def _record_step(step: TraceStep) -> None:
    collector = _current.get()
    if collector is not None:
        collector.add(step)
        return
    # No surrounding context — emit a one-step trace immediately.
    from latentspec.sdk.client import get_client

    client = get_client()
    if client is None:
        return
    trace = NormalizedTrace(
        trace_id=f"trace-{uuid.uuid4().hex[:12]}",
        agent_id=client.config.agent_id,
        timestamp=datetime.now(UTC),
        ended_at=datetime.now(UTC),
        steps=[step],
    )
    client.record(trace)


def trace_tool(
    func: Callable[..., Any] | None = None, *, name: str | None = None
) -> Callable[..., Any]:
    """Wrap a Python callable as a §3.2 tool_call step.

    The wrapped function runs normally; the step is captured side-effectfully
    via the active `StepCollector` (or shipped immediately if there isn't one).

    Works with both sync and async functions.
    """

    def _decorator(target: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or target.__name__

        if inspect.iscoroutinefunction(target):

            @functools.wraps(target)
            async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                status = "success"
                exc: BaseException | None = None
                result: Any = None
                try:
                    result = await target(*args, **kwargs)
                    return result
                except BaseException as e:  # noqa: BLE001
                    status = "error"
                    exc = e
                    raise
                finally:
                    latency_ms = max(0, int((time.perf_counter() - start) * 1000))
                    _record_step(
                        ToolCallStep(
                            tool=tool_name,
                            args=_args_to_dict(target, args, kwargs),
                            latency_ms=latency_ms,
                            result_status=status,
                            result=_safe_serialize(result),
                        )
                    )
                    if exc is not None:
                        log.debug("%s raised %r", tool_name, exc)

            return _async_wrapper

        @functools.wraps(target)
        def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            status = "success"
            exc: BaseException | None = None
            result: Any = None
            try:
                result = target(*args, **kwargs)
                return result
            except BaseException as e:  # noqa: BLE001
                status = "error"
                exc = e
                raise
            finally:
                latency_ms = max(0, int((time.perf_counter() - start) * 1000))
                _record_step(
                    ToolCallStep(
                        tool=tool_name,
                        args=_args_to_dict(target, args, kwargs),
                        latency_ms=latency_ms,
                        result_status=status,
                        result=_safe_serialize(result),
                    )
                )
                if exc is not None:
                    log.debug("%s raised %r", tool_name, exc)

        return _sync_wrapper

    if func is None:
        return _decorator
    return _decorator(func)


class trace(AbstractContextManager["trace"]):  # noqa: N801 — public API casing
    """Group multiple `@trace_tool` calls into one logical §3.2 trace.

    Usage:
        with latentspec.trace(user_input="book a flight"):
            search_flights(...)
            book_flight(...)
        # on __exit__, the assembled NormalizedTrace ships to the API.
    """

    def __init__(
        self,
        *,
        user_input: str | None = None,
        agent_id: str | None = None,
        version: str | None = None,
        user_segment: str | None = None,
    ) -> None:
        from latentspec.sdk.client import get_client

        client = get_client()
        resolved_agent_id = agent_id or (client.config.agent_id if client else "")
        self._collector = StepCollector(
            agent_id=resolved_agent_id,
            metadata=TraceMetadata(version=version, user_segment=user_segment),
        )
        if user_input is not None:
            self._collector.add(UserInputStep(content=user_input))
        self._token: contextvars.Token[StepCollector | None] | None = None
        self._final_response: str | None = None

    def __enter__(self) -> trace:
        self._token = _current.set(self._collector)
        return self

    def add_response(self, content: str) -> None:
        self._collector.add(AgentResponseStep(content=content))
        self._final_response = content

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _current.reset(self._token)
            self._token = None

        # Ship the trace
        from latentspec.sdk.client import get_client

        client = get_client()
        if client is None or not self._collector.steps:
            return
        client.record(self._collector.to_trace())


# ----- helpers ------------------------------------------------------------


def _args_to_dict(
    target: Callable[..., Any], args: tuple, kwargs: dict
) -> dict[str, Any]:
    try:
        sig = inspect.signature(target)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return {k: _safe_serialize(v) for k, v in bound.arguments.items()}
    except (TypeError, ValueError):
        return {**{f"arg_{i}": _safe_serialize(v) for i, v in enumerate(args)}, **{
            k: _safe_serialize(v) for k, v in kwargs.items()
        }}


def _safe_serialize(value: Any) -> Any:
    """Best-effort JSON-friendly conversion; truncate very large strings."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 2000 else value[:2000] + "…"
    if isinstance(value, (list, tuple)):
        return [_safe_serialize(v) for v in list(value)[:50]]
    if isinstance(value, dict):
        return {str(k): _safe_serialize(v) for k, v in list(value.items())[:50]}
    return str(value)[:2000]
