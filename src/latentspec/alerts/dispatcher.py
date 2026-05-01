"""Async alert dispatcher with per-sink retry + rate limiting.

Sinks register themselves; the dispatcher fans out events in parallel.
Retries use exponential backoff with jitter; rate limiting is per-sink and
per-agent (so a noisy agent doesn't drown out the rest of the workspace).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from latentspec.checking.base import CheckResult

log = logging.getLogger(__name__)


@dataclass
class AlertEvent:
    """One violation, ready to fan out."""

    agent_id: str
    agent_name: str
    trace_id: str
    invariant_id: str
    invariant_description: str
    severity: str
    outcome: str
    observed: str
    detected_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_check_result(
        cls,
        result: CheckResult,
        *,
        agent_id: str,
        agent_name: str,
    ) -> "AlertEvent":
        return cls(
            agent_id=agent_id,
            agent_name=agent_name,
            trace_id=result.trace_id,
            invariant_id=result.invariant_id,
            invariant_description=result.invariant_description,
            severity=result.severity.value,
            outcome=result.outcome.value,
            observed=(result.details.observed if result.details else ""),
            metadata=(result.details.extra if result.details else {}),
        )


class AlertSink(ABC):
    """Override `send`. The dispatcher handles retries + rate limiting."""

    name: str = "sink"
    max_attempts: int = 3
    base_backoff_s: float = 0.25
    rate_limit_per_min: int = 60

    @abstractmethod
    async def send(self, event: AlertEvent) -> None:
        ...


class AlertDispatcher:
    def __init__(self) -> None:
        self._sinks: list[AlertSink] = []
        self._buckets: dict[tuple[str, str], list[float]] = defaultdict(list)

    def register(self, sink: AlertSink) -> None:
        self._sinks.append(sink)

    def remove_all(self) -> None:
        self._sinks.clear()

    def sinks(self) -> list[AlertSink]:
        return list(self._sinks)

    async def dispatch(self, event: AlertEvent) -> None:
        if not self._sinks:
            return
        await asyncio.gather(*(self._send_with_retry(s, event) for s in self._sinks))

    async def _send_with_retry(self, sink: AlertSink, event: AlertEvent) -> None:
        if not self._allow(sink, event.agent_id):
            log.info("rate-limited %s for agent %s", sink.name, event.agent_id)
            return

        attempt = 0
        delay = sink.base_backoff_s
        while attempt < sink.max_attempts:
            try:
                await sink.send(event)
                return
            except Exception as e:  # noqa: BLE001
                attempt += 1
                if attempt >= sink.max_attempts:
                    log.warning(
                        "%s gave up after %d attempts: %s",
                        sink.name,
                        attempt,
                        e,
                    )
                    return
                # Exponential backoff with full jitter
                jitter = random.uniform(0, delay)
                await asyncio.sleep(delay + jitter)
                delay *= 2

    def _allow(self, sink: AlertSink, agent_id: str) -> bool:
        bucket = self._buckets[(sink.name, agent_id)]
        now = time.time()
        cutoff = now - 60.0
        # drop expired hits
        while bucket and bucket[0] < cutoff:
            bucket.pop(0)
        if len(bucket) >= sink.rate_limit_per_min:
            return False
        bucket.append(now)
        return True


_singleton: AlertDispatcher | None = None


def get_dispatcher() -> AlertDispatcher:
    global _singleton
    if _singleton is None:
        _singleton = AlertDispatcher()
    return _singleton
