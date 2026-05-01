"""In-process Prometheus-compatible metrics.

This module avoids a hard dep on `prometheus_client`; values are kept in
process and rendered to text on demand (suitable for `/metrics` scraping).
For multi-process deploys the rendered text is union-able across workers.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


_lock = threading.Lock()


@dataclass
class _Counter:
    value: float = 0.0


@dataclass
class _Histogram:
    bucket_bounds: tuple[float, ...] = (
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    )
    buckets: list[float] = field(default_factory=lambda: [0.0] * 13)
    sum: float = 0.0
    count: int = 0


_counters: dict[tuple[str, tuple[tuple[str, str], ...]], _Counter] = defaultdict(_Counter)
_histograms: dict[tuple[str, tuple[tuple[str, str], ...]], _Histogram] = defaultdict(_Histogram)


def _label_tuple(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((labels or {}).items()))


def counter(name: str, *, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
    key = (name, _label_tuple(labels))
    with _lock:
        _counters[key].value += value


def histogram(
    name: str,
    value_seconds: float,
    *,
    labels: dict[str, str] | None = None,
) -> None:
    key = (name, _label_tuple(labels))
    with _lock:
        h = _histograms[key]
        h.sum += value_seconds
        h.count += 1
        for i, bound in enumerate(h.bucket_bounds):
            if value_seconds <= bound:
                h.buckets[i] += 1
        # implicit +Inf bucket: every observation always lands in it (count above)


@contextmanager
def timer(name: str, *, labels: dict[str, str] | None = None) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        histogram(name, time.perf_counter() - start, labels=labels)


def metrics_text() -> str:
    """Render the current snapshot in Prometheus exposition format."""
    lines: list[str] = []
    with _lock:
        for (name, labels), c in _counters.items():
            lines.append(f"# TYPE {name} counter")
            label_str = _format_labels(labels)
            lines.append(f"{name}{label_str} {c.value}")
        for (name, labels), h in _histograms.items():
            lines.append(f"# TYPE {name} histogram")
            label_str = _format_labels(labels)
            for bound, count in zip(h.bucket_bounds, h.buckets, strict=True):
                lines.append(
                    f'{name}_bucket{_with_le(label_str, bound)} {count}'
                )
            lines.append(f"{name}_bucket{_with_le(label_str, '+Inf')} {h.count}")
            lines.append(f"{name}_sum{label_str} {h.sum}")
            lines.append(f"{name}_count{label_str} {h.count}")
    return "\n".join(lines) + "\n"


def _format_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    inside = ",".join(f'{k}="{v}"' for k, v in labels)
    return f"{{{inside}}}"


def _with_le(label_str: str, le: object) -> str:
    if not label_str:
        return f'{{le="{le}"}}'
    return label_str[:-1] + f',le="{le}"}}'
