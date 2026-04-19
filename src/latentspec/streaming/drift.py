"""Online drift detection — Page-Hinkley + CUSUM (§4.1 Growth phase).

Mining produces invariants from a *snapshot* of trace data. Between mining
runs the agent's behavior can change — a prompt update, a model upgrade, a
new tool — and the active rule set quietly stops matching reality. The
streaming detector calls into this module on every check result so we
catch the drift as it happens rather than at the next mining cadence.

Two sequential change-detection algorithms:

- **Page-Hinkley** [Page 1954, Hinkley 1971]: maintains a one-sided cumsum
  of `(observation - mean) - delta` and fires when `cumsum - min(cumsum)`
  exceeds `threshold`. Detects monotone drops in the running mean (i.e.
  rule pass-rate dropping). The de-facto standard sequential test in
  streaming-ML libraries (River, scikit-multiflow).

- **CUSUM** [Page 1954]: paired upper/lower cumulative sums centred on a
  reference value. Symmetric — catches both increases (false-positive
  spike) and decreases (rule starting to fail). Useful for the
  warn-rate metric where either direction matters.

Per-`(agent_id, invariant_id)` state lives in a `DriftRegistry` that the
streaming detector references on every check. Detected drift fires an
alert via the standard dispatcher with `severity=high` and a payload that
includes the running mean, cumsum, and observed window.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

log = logging.getLogger(__name__)


# ---------------- Page-Hinkley --------------------------------------------


@dataclass
class PageHinkleyDetector:
    """One-sided Page-Hinkley test for mean drops.

    Args:
        threshold: ``λ`` — fire when the deviation exceeds this. Higher =
            more conservative (fewer false alarms, slower detection).
            Standard literature defaults: ``50`` for raw counters,
            ``5–25`` for binary observations after EWMA smoothing.
        delta: tolerance above the running mean before observations count
            against the cumsum.
        alpha: EWMA forgetting factor for the running mean estimate.
        min_samples: don't fire alarms before this many observations
            have been seen (warmup).
    """

    threshold: float = 8.0
    delta: float = 0.005
    alpha: float = 0.05
    min_samples: int = 30

    mean: float = 0.0
    cumsum: float = 0.0
    min_cumsum: float = 0.0
    n: int = 0
    fired: bool = False
    last_value: float | None = None
    last_alarm_at: float | None = None

    def update(self, value: float) -> bool:
        """Feed a single observation. Returns True the first call after
        a drift alarm fires; returns False otherwise."""
        self.n += 1
        self.last_value = value
        # EWMA-smoothed running mean
        if self.n == 1:
            self.mean = value
        else:
            self.mean = (1 - self.alpha) * self.mean + self.alpha * value
        self.cumsum += (self.mean - value) - self.delta
        if self.cumsum < self.min_cumsum:
            self.min_cumsum = self.cumsum

        if self.n < self.min_samples:
            return False

        ph = self.cumsum - self.min_cumsum
        if ph > self.threshold and not self.fired:
            self.fired = True
            self.last_alarm_at = time.time()
            return True
        return False

    def reset(self) -> None:
        self.mean = 0.0
        self.cumsum = 0.0
        self.min_cumsum = 0.0
        self.n = 0
        self.fired = False
        self.last_value = None


# ---------------- CUSUM ---------------------------------------------------


@dataclass
class CusumDetector:
    """Two-sided CUSUM with reference value `target`.

    Args:
        target: reference value. Each observation contributes
            ``s+ += max(0, (x - target) - k)`` (upper) and
            ``s- += min(0, (x - target) + k)`` (lower).
        slack: ``k`` — half the smallest shift we want to detect quickly.
        threshold: ``h`` — fire on either bound crossing this.
        min_samples: warmup count before alarms can fire.
    """

    target: float = 0.0
    slack: float = 0.5
    threshold: float = 5.0
    min_samples: int = 20

    s_pos: float = 0.0
    s_neg: float = 0.0
    n: int = 0
    fired: bool = False
    direction: str | None = None
    last_alarm_at: float | None = None

    def update(self, value: float) -> bool:
        self.n += 1
        deviation = value - self.target
        self.s_pos = max(0.0, self.s_pos + deviation - self.slack)
        self.s_neg = min(0.0, self.s_neg + deviation + self.slack)

        if self.n < self.min_samples:
            return False
        if not self.fired:
            if self.s_pos > self.threshold:
                self.fired = True
                self.direction = "up"
                self.last_alarm_at = time.time()
                return True
            if -self.s_neg > self.threshold:
                self.fired = True
                self.direction = "down"
                self.last_alarm_at = time.time()
                return True
        return False

    def reset(self) -> None:
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.n = 0
        self.fired = False
        self.direction = None


# ---------------- Per-rule registry --------------------------------------


@dataclass
class DriftEvent:
    agent_id: str
    invariant_id: str
    detector: str  # "page-hinkley" | "cusum"
    direction: str | None
    n_observations: int
    running_mean: float
    cumulative_value: float
    detected_at: float = field(default_factory=time.time)


class DriftRegistry:
    """Process-wide registry mapping (agent_id, invariant_id) -> detector pair.

    Streaming detector calls `observe(agent_id, invariant_id, passed: bool)`
    on every check. We track two metrics in parallel:
      - pass-rate via Page-Hinkley (catches degradation)
      - pass-rate via CUSUM with target=baseline (catches symmetric shift)

    A `DriftEvent` is emitted the first time either fires; subsequent
    observations are still recorded but don't re-fire until reset.
    """

    def __init__(
        self,
        *,
        ph_threshold: float = 8.0,
        ph_delta: float = 0.005,
        cusum_slack: float = 0.05,
        cusum_threshold: float = 4.0,
        observation_window: int = 256,
    ) -> None:
        self._ph_kwargs = {"threshold": ph_threshold, "delta": ph_delta}
        self._cusum_kwargs = {
            "slack": cusum_slack,
            "threshold": cusum_threshold,
        }
        self._window = observation_window
        self._lock = threading.Lock()
        self._page_hinkley: dict[tuple[str, str], PageHinkleyDetector] = {}
        self._cusum: dict[tuple[str, str], CusumDetector] = {}
        self._observations: dict[tuple[str, str], deque[float]] = {}
        self._baseline: dict[tuple[str, str], float] = {}

    def observe(
        self, agent_id: str, invariant_id: str, passed: bool
    ) -> list[DriftEvent]:
        key = (agent_id, invariant_id)
        value = 1.0 if passed else 0.0
        events: list[DriftEvent] = []

        with self._lock:
            ph = self._page_hinkley.setdefault(
                key, PageHinkleyDetector(**self._ph_kwargs)
            )
            cs = self._cusum.setdefault(key, CusumDetector(**self._cusum_kwargs))
            obs = self._observations.setdefault(
                key, deque(maxlen=self._window)
            )
            obs.append(value)

            # Use the rolling mean as the CUSUM target until we've seen enough,
            # then freeze the baseline.
            if key not in self._baseline and len(obs) >= 64:
                self._baseline[key] = sum(obs) / max(1, len(obs))
                cs.target = self._baseline[key]
            elif key not in self._baseline:
                cs.target = sum(obs) / max(1, len(obs))

            ph_fired = ph.update(value)
            cs_fired = cs.update(value)

            if ph_fired:
                events.append(
                    DriftEvent(
                        agent_id=agent_id,
                        invariant_id=invariant_id,
                        detector="page-hinkley",
                        direction="down",
                        n_observations=ph.n,
                        running_mean=ph.mean,
                        cumulative_value=ph.cumsum - ph.min_cumsum,
                    )
                )
            if cs_fired:
                events.append(
                    DriftEvent(
                        agent_id=agent_id,
                        invariant_id=invariant_id,
                        detector="cusum",
                        direction=cs.direction,
                        n_observations=cs.n,
                        running_mean=sum(obs) / max(1, len(obs)),
                        cumulative_value=cs.s_pos if cs.direction == "up" else cs.s_neg,
                    )
                )
        return events

    def reset(
        self, agent_id: str | None = None, invariant_id: str | None = None
    ) -> None:
        """Reset all detectors (call after re-mining), or just one
        (agent_id, invariant_id) pair."""
        with self._lock:
            if agent_id is None and invariant_id is None:
                self._page_hinkley.clear()
                self._cusum.clear()
                self._observations.clear()
                self._baseline.clear()
                return
            keys = [
                k
                for k in list(self._page_hinkley)
                if (agent_id is None or k[0] == agent_id)
                and (invariant_id is None or k[1] == invariant_id)
            ]
            for k in keys:
                self._page_hinkley.pop(k, None)
                self._cusum.pop(k, None)
                self._observations.pop(k, None)
                self._baseline.pop(k, None)

    def stats(self) -> dict[str, dict[str, float]]:
        """Snapshot for `/metrics` / dashboards."""
        with self._lock:
            out: dict[str, dict[str, float]] = {}
            for key, ph in self._page_hinkley.items():
                kid = f"{key[0]}::{key[1]}"
                out[kid] = {
                    "ph_n": float(ph.n),
                    "ph_mean": ph.mean,
                    "ph_cumsum_excursion": ph.cumsum - ph.min_cumsum,
                    "ph_fired": float(ph.fired),
                }
            return out


_singleton: DriftRegistry | None = None


def get_drift_registry() -> DriftRegistry:
    global _singleton
    if _singleton is None:
        _singleton = DriftRegistry()
    return _singleton


def configure_for_test(registry: DriftRegistry | None) -> None:
    global _singleton
    _singleton = registry
