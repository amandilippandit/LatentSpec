"""SDK client and global init() / record_trace() / flush() functions.

Trace shipment runs on a background thread so wrapping a tool function
adds <1ms to the host call. We POST to `/traces` with `format="normalized"`
since the SDK constructs §3.2 traces directly.
"""

from __future__ import annotations

import atexit
import logging
import os
import queue
import threading
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import httpx

from latentspec.schemas.trace import NormalizedTrace

log = logging.getLogger(__name__)


@dataclass
class SDKConfig:
    api_key: str = ""
    agent_id: str = ""
    api_base: str = "https://api.latentspec.dev"
    timeout: float = 5.0
    queue_max: int = 1024
    flush_threshold: int = 16
    flush_interval_seconds: float = 5.0
    enabled: bool = True
    extra_headers: dict[str, str] = field(default_factory=dict)


class LatentSpecClient:
    """Background-flushing trace shipper.

    Designed so that `client.record(trace)` returns immediately and the
    network I/O happens off the request path.
    """

    def __init__(self, config: SDKConfig) -> None:
        self._config = config
        self._queue: queue.Queue[NormalizedTrace | None] = queue.Queue(
            maxsize=config.queue_max
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._client: httpx.Client | None = None

        if config.enabled and config.api_key and config.api_base:
            self._client = httpx.Client(
                base_url=config.api_base.rstrip("/"),
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "User-Agent": "latentspec-python/0.1",
                    **config.extra_headers,
                },
                timeout=config.timeout,
            )
            self._thread = threading.Thread(
                target=self._run, name="latentspec-flusher", daemon=True
            )
            self._thread.start()

    @property
    def config(self) -> SDKConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return self._client is not None and not self._stop.is_set()

    def record(self, trace: NormalizedTrace) -> None:
        if not self.enabled:
            return
        # Sampling decision (deterministic on trace_id by default).
        from latentspec.sdk.sampling import get_default_sampler

        if not get_default_sampler().keep(trace):
            return

        # PII redaction sweep before the trace leaves the process.
        from latentspec.sdk.redaction import get_default_redactor

        redactor = get_default_redactor()
        if redactor.enabled:
            trace = _redact_trace(trace, redactor)

        try:
            self._queue.put_nowait(trace)
        except queue.Full:
            log.warning("latentspec SDK queue full; dropping trace %s", trace.trace_id)

    def flush(self, timeout: float = 5.0) -> None:
        """Block until everything currently queued has been shipped."""
        if not self.enabled:
            return
        sentinel = threading.Event()

        def _wait() -> None:
            self._queue.join()
            sentinel.set()

        threading.Thread(target=_wait, daemon=True).start()
        sentinel.wait(timeout=timeout)

    def shutdown(self, timeout: float = 5.0) -> None:
        if self._stop.is_set():
            return
        self.flush(timeout=timeout)
        self._stop.set()
        # Wake the worker
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._client is not None:
            self._client.close()
            self._client = None

    # ----- internals -------------------------------------------------------

    def _run(self) -> None:
        batch: list[NormalizedTrace] = []
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=self._config.flush_interval_seconds)
            except queue.Empty:
                if batch:
                    self._send(batch)
                    batch = []
                continue
            if item is None:
                self._queue.task_done()
                break
            batch.append(item)
            self._queue.task_done()
            if len(batch) >= self._config.flush_threshold:
                self._send(batch)
                batch = []
        if batch:
            self._send(batch)

    def _send(self, traces: list[NormalizedTrace]) -> None:
        if self._client is None or not traces:
            return
        try:
            payload = {
                "traces": [
                    {
                        "agent_id": str(self._coerce_agent_id(self._config.agent_id)),
                        "format": "normalized",
                        "payload": t.model_dump(mode="json"),
                        "version_tag": t.metadata.version,
                    }
                    for t in traces
                ]
            }
            resp = self._client.post("/traces/batch", json=payload)
            if resp.status_code >= 400:
                log.warning(
                    "latentspec /traces/batch returned %s: %s",
                    resp.status_code,
                    resp.text[:200],
                )
        except Exception as e:
            log.warning("latentspec trace shipment failed: %s", e)

    @staticmethod
    def _coerce_agent_id(agent_id: str) -> UUID | str:
        try:
            return UUID(agent_id)
        except (ValueError, TypeError):
            return agent_id


# ---------------- module-level singleton API (the public surface) ---------

_client: LatentSpecClient | None = None
_lock = threading.Lock()


def init(
    *,
    api_key: str | None = None,
    agent_id: str | None = None,
    api_base: str | None = None,
    enabled: bool = True,
    timeout: float = 5.0,
) -> LatentSpecClient:
    """Bootstrap the SDK. Idempotent — calling twice replaces the client."""
    global _client
    config = SDKConfig(
        api_key=api_key or os.environ.get("LATENTSPEC_API_KEY", ""),
        agent_id=agent_id or os.environ.get("LATENTSPEC_AGENT_ID", ""),
        api_base=api_base or os.environ.get(
            "LATENTSPEC_API_BASE", "https://api.latentspec.dev"
        ),
        enabled=enabled,
        timeout=timeout,
    )
    with _lock:
        if _client is not None:
            _client.shutdown(timeout=1.0)
        _client = LatentSpecClient(config)
        atexit.register(_client.shutdown)
    return _client


def get_client() -> LatentSpecClient | None:
    return _client


def is_initialized() -> bool:
    return _client is not None and _client.enabled


def record_trace(trace: NormalizedTrace) -> None:
    """Ship a fully-formed §3.2 trace."""
    if _client is None:
        return
    _client.record(trace)


def flush(timeout: float = 5.0) -> None:
    if _client is not None:
        _client.flush(timeout=timeout)


def shutdown(timeout: float = 5.0) -> None:
    global _client
    with _lock:
        if _client is None:
            return
        _client.shutdown(timeout=timeout)
        _client = None


def configure_for_test(client: LatentSpecClient | None) -> None:
    """Test helper — inject a custom (or fake) client."""
    global _client
    with _lock:
        _client = client


def _redact_trace(trace: NormalizedTrace, redactor: Any) -> NormalizedTrace:
    """Return a copy of `trace` with redaction applied to every string field."""
    new_steps = []
    for step in trace.steps:
        data = step.model_dump()
        for key in ("content", "args", "result"):
            if key in data:
                data[key] = redactor.redact_value(key, data[key])
        new_steps.append(type(step).model_validate(data))
    return trace.model_copy(update={"steps": new_steps})
