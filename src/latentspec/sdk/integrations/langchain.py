"""LangChain / LangGraph integration (§5.2 P0).

Two surfaces:
  - `wrap(agent)` — returns a thin wrapper that runs the agent normally and
    captures every callback event into a §3.2 trace.
  - `LatentSpecCallbackHandler` — for users who already manage their own
    callback list (LangGraph, custom Runnables).

The handler stores partial state on a thread-safe dict keyed by run_id, so
nested chains/tools resolve correctly. We deliberately avoid importing
langchain at module-import time so users without LangChain don't pay the
weight; callbacks accept dict-shaped events too for testing purposes.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from latentspec.schemas.trace import (
    AgentResponseStep,
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)

log = logging.getLogger(__name__)


class LatentSpecCallbackHandler:
    """LangChain BaseCallbackHandler-compatible event capture.

    Implements the union of methods used across the LangChain JS/Python
    callback APIs. Methods do nothing dangerous when LangChain is not
    actually installed — they just append to internal state.
    """

    def __init__(self, *, agent_id: str | None = None) -> None:
        from latentspec.sdk.client import get_client

        client = get_client()
        self._agent_id = agent_id or (client.config.agent_id if client else "")
        self._lock = threading.Lock()
        self._tool_starts: dict[str, tuple[str, dict, float]] = {}
        self._chain: NormalizedTrace | None = None
        self._steps: list[Any] = []
        self._user_input: str | None = None
        self._final_response: str | None = None
        self._started_at = datetime.now(UTC)
        self._metadata = TraceMetadata()

    # ----- chain (top-level run) ------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if self._user_input is not None:
            return
        self._started_at = datetime.now(UTC)
        text = self._extract_user_input(inputs)
        if text is not None:
            self._user_input = text
            self._steps.append(UserInputStep(content=text))

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        text = self._extract_response(outputs)
        if text is not None and self._final_response is None:
            self._final_response = text
            self._steps.append(AgentResponseStep(content=text))

    def on_chain_error(
        self, error: BaseException, **kwargs: Any
    ) -> None:
        log.debug("LangChain chain error: %r", error)

    # ----- tool ------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: Any,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        tool_name = (serialized or {}).get("name") or "unknown_tool"
        args: dict[str, Any]
        if isinstance(input_str, dict):
            args = input_str
        elif isinstance(input_str, str):
            args = {"input": input_str}
        else:
            args = {"input": str(input_str)}
        with self._lock:
            self._tool_starts[str(run_id)] = (
                tool_name,
                args,
                time.perf_counter(),
            )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            entry = self._tool_starts.pop(str(run_id), None)
        if entry is None:
            return
        name, args, start = entry
        latency_ms = max(0, int((time.perf_counter() - start) * 1000))
        self._steps.append(
            ToolCallStep(
                tool=name,
                args=args,
                latency_ms=latency_ms,
                result_status="success",
                result=_safe(output),
            )
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            entry = self._tool_starts.pop(str(run_id), None)
        if entry is None:
            return
        name, args, start = entry
        latency_ms = max(0, int((time.perf_counter() - start) * 1000))
        self._steps.append(
            ToolCallStep(
                tool=name,
                args=args,
                latency_ms=latency_ms,
                result_status="error",
                result={"error": repr(error)[:500]},
            )
        )

    # ----- llm (intermediate calls) ---------------------------------------

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        # We don't currently emit a step for intermediate LLM calls — the
        # information is already implied by surrounding chain/tool events.
        return

    # ----- emit ------------------------------------------------------------

    def emit(self, *, version: str | None = None) -> NormalizedTrace | None:
        """Build the §3.2 trace and ship it. Idempotent."""
        if not self._steps:
            return None
        trace = NormalizedTrace(
            trace_id=f"trace-{uuid.uuid4().hex[:12]}",
            agent_id=self._agent_id,
            timestamp=self._started_at,
            ended_at=datetime.now(UTC),
            steps=list(self._steps),
            metadata=TraceMetadata(version=version) if version else self._metadata,
        )

        from latentspec.sdk.client import get_client

        client = get_client()
        if client is not None:
            client.record(trace)

        # Reset for the next run.
        self._steps.clear()
        self._user_input = None
        self._final_response = None
        return trace

    # ----- helpers ---------------------------------------------------------

    @staticmethod
    def _extract_user_input(inputs: dict[str, Any] | None) -> str | None:
        if not inputs:
            return None
        for key in ("input", "question", "query", "prompt"):
            v = inputs.get(key)
            if v:
                if isinstance(v, str):
                    return v
                if isinstance(v, list) and v and isinstance(v[-1], dict):
                    return str(v[-1].get("content") or v[-1])
                return str(v)
        return None

    @staticmethod
    def _extract_response(outputs: dict[str, Any] | None) -> str | None:
        if not outputs:
            return None
        for key in ("output", "answer", "result", "response", "content"):
            v = outputs.get(key)
            if v:
                if isinstance(v, str):
                    return v
                if isinstance(v, dict):
                    return str(v.get("content") or v.get("text") or v)
                return str(v)
        return None


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 2000 else value[:2000] + "…"
    return str(value)[:2000]


def wrap(agent: Any, *, agent_id: str | None = None) -> Any:
    """Attach a LatentSpec callback handler to a LangChain agent.

    Returns a thin wrapper exposing `.invoke()` / `.ainvoke()` that flushes
    a trace on each call. Falls back gracefully when given an object that
    doesn't quite match the LangChain Runnable shape.
    """
    handler = LatentSpecCallbackHandler(agent_id=agent_id)

    class _Wrapped:
        __slots__ = ("_inner", "_handler")

        def __init__(self) -> None:
            self._inner = agent
            self._handler = handler

        @property
        def callback_handler(self) -> LatentSpecCallbackHandler:
            return self._handler

        def invoke(self, *args: Any, **kwargs: Any) -> Any:
            cfg = kwargs.get("config") or {}
            cbs = list(cfg.get("callbacks") or [])
            cbs.append(self._handler)
            cfg["callbacks"] = cbs
            kwargs["config"] = cfg
            try:
                return self._inner.invoke(*args, **kwargs)
            finally:
                self._handler.emit()

        async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
            cfg = kwargs.get("config") or {}
            cbs = list(cfg.get("callbacks") or [])
            cbs.append(self._handler)
            cfg["callbacks"] = cbs
            kwargs["config"] = cfg
            try:
                ainvoke = getattr(self._inner, "ainvoke", None)
                if ainvoke is None:
                    return self._inner.invoke(*args, **kwargs)
                return await ainvoke(*args, **kwargs)
            finally:
                self._handler.emit()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    return _Wrapped()
