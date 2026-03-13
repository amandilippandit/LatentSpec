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
