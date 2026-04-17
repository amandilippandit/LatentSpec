"""Hot invariant cache — Redis or in-process.

Invariant sets per agent are pulled into memory once and revalidated via
a TTL + an explicit invalidation pubsub channel (`latentspec:inv:invalidate`).

The in-process cache is the default for local dev / single-node deploys; the
Redis cache is the default for production.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from latentspec.checking.base import InvariantSpec
from latentspec.config import get_settings
from latentspec.models.invariant import InvariantStatus, InvariantType, Severity

log = logging.getLogger(__name__)


_INVALIDATE_CHANNEL = "latentspec:inv:invalidate"
_KEY_PREFIX = "ls:inv:"


def _spec_to_dict(spec: InvariantSpec) -> dict[str, Any]:
    return {
        "id": spec.id,
        "type": spec.type.value,
        "description": spec.description,
        "formal_rule": spec.formal_rule,
        "severity": spec.severity.value,
        "params": dict(spec.params),
    }


def _spec_from_dict(payload: dict[str, Any]) -> InvariantSpec:
    return InvariantSpec(
        id=str(payload["id"]),
        type=InvariantType(payload["type"]),
        description=payload["description"],
        formal_rule=payload.get("formal_rule") or "",
        severity=Severity(payload.get("severity", "medium")),
        params=dict(payload.get("params") or {}),
    )


@dataclass
class CacheEntry:
    """In-memory representation of one agent's active invariant set."""

    invariants: list[InvariantSpec]
    fetched_at: float = field(default_factory=time.time)
    version: int = 0


class HotInvariantCache(ABC):
    """Interface for hot caches. Sync get_local() for the streaming detector,
    async get_or_load() for cold-start population."""

    ttl_seconds: float = 30.0

    @abstractmethod
    def get_local(self, agent_id: str) -> list[InvariantSpec] | None: ...

    @abstractmethod
    async def get_or_load(
        self, agent_id: str, *, loader: "_AsyncLoader"
    ) -> list[InvariantSpec]: ...

    @abstractmethod
    async def invalidate(self, agent_id: str) -> None: ...

    @abstractmethod
    async def warm(self, agent_id: str, invariants: list[InvariantSpec]) -> None: ...


# Function signature: async loader that produces an authoritative set from PG
_AsyncLoader = "Callable[[str], Awaitable[list[InvariantSpec]]]"


class InMemoryCache(HotInvariantCache):
    """Fallback cache used when no Redis is configured. Process-local only."""

    def __init__(self, *, ttl_seconds: float = 30.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

    def get_local(self, agent_id: str) -> list[InvariantSpec] | None:
        entry = self._store.get(agent_id)
        if entry is None:
            return None
        if time.time() - entry.fetched_at > self.ttl_seconds:
            return None
        return entry.invariants

    async def get_or_load(self, agent_id, *, loader):
        local = self.get_local(agent_id)
        if local is not None:
            return local
        async with self._lock:
            local = self.get_local(agent_id)
            if local is not None:
                return local
            invariants = await loader(agent_id)
            self._store[agent_id] = CacheEntry(invariants=invariants)
            return invariants

    async def invalidate(self, agent_id: str) -> None:
        self._store.pop(agent_id, None)

    async def warm(self, agent_id: str, invariants: list[InvariantSpec]) -> None:
        self._store[agent_id] = CacheEntry(invariants=list(invariants))
