"""Structured logging + Prometheus-style metrics.

This module is the boot-time call site for ops-grade telemetry. Wire it in
once at process start (`configure_logging()`) and every other module's
`logging.getLogger(__name__).info(...)` produces JSON suitable for
Datadog/Loki/Honeycomb ingestion.
"""

from latentspec.observability.logging import configure_logging
from latentspec.observability.metrics import (
    counter,
    histogram,
    metrics_text,
    timer,
)

__all__ = ["configure_logging", "counter", "histogram", "metrics_text", "timer"]
