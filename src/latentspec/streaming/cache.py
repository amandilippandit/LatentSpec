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


class RedisCache(HotInvariantCache):
    """Redis-backed cache with pubsub-driven invalidation across nodes."""

    def __init__(
        self,
        *,
        url: str | None = None,
        ttl_seconds: float = 30.0,
    ) -> None:
        import redis.asyncio as redis_async

        self.ttl_seconds = ttl_seconds
        self._url = url or get_settings().redis_url
        self._async_client = redis_async.from_url(self._url, decode_responses=True)
        self._local: dict[str, CacheEntry] = {}
        self._listener_task: asyncio.Task[None] | None = None

    def _key(self, agent_id: str) -> str:
        return f"{_KEY_PREFIX}{agent_id}"

    def get_local(self, agent_id: str) -> list[InvariantSpec] | None:
        entry = self._local.get(agent_id)
        if entry is None:
            return None
        if time.time() - entry.fetched_at > self.ttl_seconds:
            return None
        return entry.invariants

    async def get_or_load(self, agent_id, *, loader):
        local = self.get_local(agent_id)
        if local is not None:
            return local

        # Try Redis
        try:
            payload = await self._async_client.get(self._key(agent_id))
        except Exception as e:  # noqa: BLE001
            log.warning("redis read failed: %s", e)
            payload = None

        if payload:
            try:
                items = json.loads(payload)
                invariants = [_spec_from_dict(p) for p in items]
                self._local[agent_id] = CacheEntry(invariants=invariants)
                return invariants
            except json.JSONDecodeError as e:
                log.warning("malformed cache payload for %s: %s", agent_id, e)

        # Cold load from authoritative source
        invariants = await loader(agent_id)
        await self.warm(agent_id, invariants)
        return invariants

    async def warm(self, agent_id: str, invariants: list[InvariantSpec]) -> None:
        self._local[agent_id] = CacheEntry(invariants=list(invariants))
        try:
            await self._async_client.set(
                self._key(agent_id),
                json.dumps([_spec_to_dict(s) for s in invariants]),
                ex=int(self.ttl_seconds * 4),
            )
        except Exception as e:  # noqa: BLE001
            log.warning("redis warm failed: %s", e)

    async def invalidate(self, agent_id: str) -> None:
        self._local.pop(agent_id, None)
        try:
            await self._async_client.delete(self._key(agent_id))
            await self._async_client.publish(_INVALIDATE_CHANNEL, agent_id)
        except Exception as e:  # noqa: BLE001
            log.warning("redis invalidate failed: %s", e)

    async def start_listener(self) -> None:
        """Subscribe to invalidation pubsub. Runs forever; call once at boot."""
        if self._listener_task is not None and not self._listener_task.done():
            return

        async def _run() -> None:
            try:
                pubsub = self._async_client.pubsub()
                await pubsub.subscribe(_INVALIDATE_CHANNEL)
                async for msg in pubsub.listen():
                    if msg.get("type") != "message":
                        continue
                    agent_id = msg.get("data")
                    if isinstance(agent_id, bytes):
                        agent_id = agent_id.decode()
                    if isinstance(agent_id, str):
                        self._local.pop(agent_id, None)
            except Exception as e:  # noqa: BLE001
                log.warning("redis listener stopped: %s", e)

        self._listener_task = asyncio.create_task(_run())


_singleton: HotInvariantCache | None = None


def get_cache() -> HotInvariantCache:
    """Return a process-wide cache. Redis if reachable; in-process fallback."""
    global _singleton
    if _singleton is not None:
        return _singleton
    settings = get_settings()
    if settings.redis_url and not settings.redis_url.startswith("memory://"):
        try:
            _singleton = RedisCache(url=settings.redis_url)
            return _singleton
        except Exception as e:  # noqa: BLE001 — fall back gracefully
            log.warning("Redis cache init failed (%s); using in-memory fallback", e)
    _singleton = InMemoryCache()
    return _singleton


def configure_cache_for_test(cache: HotInvariantCache | None) -> None:
    global _singleton
    _singleton = cache
