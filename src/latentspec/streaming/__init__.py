"""§4.1 Real-time streaming detector + Redis hot invariant cache.

The hot path:
  1. Trace arrives at the API (or is recorded by the SDK).
  2. The streaming detector pulls the active invariant set from Redis
     (sub-millisecond).
  3. Rule-based checkers evaluate every applicable invariant against the
     trace (sub-100ms total per §4.1).
  4. Violations are emitted to the alert dispatcher and persisted as
     `violations` rows.

We bypass Postgres on the read side. The cache is invalidated on every
mining run + every PATCH /invariants/{id} via a publisher channel.
"""

from latentspec.streaming.cache import (
    HotInvariantCache,
    InMemoryCache,
    RedisCache,
    get_cache,
)
from latentspec.streaming.detector import (
    DetectionStats,
    StreamingDetector,
    StreamingResult,
)
from latentspec.streaming.drift import (
    CusumDetector,
    DriftEvent,
    DriftRegistry,
    PageHinkleyDetector,
    get_drift_registry,
)

__all__ = [
    "CusumDetector",
    "DetectionStats",
    "DriftEvent",
    "DriftRegistry",
    "HotInvariantCache",
    "InMemoryCache",
    "PageHinkleyDetector",
    "RedisCache",
    "StreamingDetector",
    "StreamingResult",
    "get_cache",
    "get_drift_registry",
]
