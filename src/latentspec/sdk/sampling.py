"""Sampling at the SDK boundary.

Production agents at scale produce millions of traces/day. This module
offers three orthogonal sampling strategies any of which can be combined:

  - rate            — uniform random sampling at a configured probability
  - adaptive        — keep all error/long-latency/segment-rare traces; rate-sample the rest
  - tail-on-error   — always keep failed traces; rate-sample successes

Hash-based deterministic sampling (over `trace_id`) makes the decision stable
across retries, so a trace either reaches the API every time or never does.
"""

from __future__ import annotations

import enum
import hashlib
import random
from dataclasses import dataclass

from latentspec.schemas.trace import NormalizedTrace, ToolCallStep


class SamplingStrategy(str, enum.Enum):
    NONE = "none"
    RATE = "rate"
    ADAPTIVE = "adaptive"
    TAIL_ON_ERROR = "tail_on_error"


@dataclass
class Sampler:
    strategy: SamplingStrategy = SamplingStrategy.NONE
    rate: float = 1.0
    keep_errors: bool = True
    keep_segment_rare: bool = True
    long_latency_ms: int = 1000
    deterministic: bool = True

    def keep(self, trace: NormalizedTrace) -> bool:
        if self.strategy == SamplingStrategy.NONE:
            return True

        # Always keep errors when configured (cheap signal, hard to reproduce)
        if self.keep_errors and self._has_error(trace):
            return True

        # Adaptive: keep long-latency / rare-segment traces
        if self.strategy == SamplingStrategy.ADAPTIVE:
            if self._has_long_latency(trace):
                return True
            if self.keep_segment_rare and self._is_rare_segment(trace):
                return True

        # Tail-on-error: only errors plus a sampled subset of successes
        if self.strategy == SamplingStrategy.TAIL_ON_ERROR:
            return self._sample(trace)

        # Default: rate-based
        return self._sample(trace)

    def _sample(self, trace: NormalizedTrace) -> bool:
        if self.rate >= 1.0:
            return True
        if self.rate <= 0.0:
            return False
        if self.deterministic:
            digest = hashlib.sha1(trace.trace_id.encode("utf-8")).digest()
            roll = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
        else:
            roll = random.random()
        return roll < self.rate

    @staticmethod
    def _has_error(trace: NormalizedTrace) -> bool:
        for step in trace.steps:
            if isinstance(step, ToolCallStep) and (
                (step.result_status or "success") != "success"
            ):
                return True
        return False

    def _has_long_latency(self, trace: NormalizedTrace) -> bool:
        for step in trace.steps:
            if isinstance(step, ToolCallStep) and (step.latency_ms or 0) >= self.long_latency_ms:
                return True
        return False

    @staticmethod
    def _is_rare_segment(trace: NormalizedTrace) -> bool:
        seg = trace.metadata.user_segment
        # The SDK doesn't track segment frequency; this is a hook future
        # implementations populate from a Redis-backed sketch. Default: keep.
        return seg is not None


_default = Sampler()


def get_default_sampler() -> Sampler:
    return _default


def configure(
    *,
    strategy: SamplingStrategy | str = SamplingStrategy.NONE,
    rate: float = 1.0,
    keep_errors: bool = True,
    long_latency_ms: int = 1000,
) -> None:
    global _default
    _default.strategy = SamplingStrategy(strategy) if isinstance(strategy, str) else strategy
    _default.rate = max(0.0, min(1.0, rate))
    _default.keep_errors = keep_errors
    _default.long_latency_ms = long_latency_ms
