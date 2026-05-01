"""PagerDuty Events API v2 sink — only critical/high severity by default."""

from __future__ import annotations

import httpx

from latentspec.alerts.dispatcher import AlertEvent, AlertSink


_SEV_MAP = {
    "critical": "critical",
    "high":     "error",
    "medium":   "warning",
    "low":      "info",
}


class PagerDutySink(AlertSink):
    name = "pagerduty"

    def __init__(
        self,
        routing_key: str,
        *,
        min_severity: str = "high",
        endpoint: str = "https://events.pagerduty.com/v2/enqueue",
    ) -> None:
        self._routing_key = routing_key
        self._min_severity = min_severity
        self._endpoint = endpoint

    def _meets_min(self, severity: str) -> bool:
        order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        return order.get(severity, 0) >= order.get(self._min_severity, 2)

    async def send(self, event: AlertEvent) -> None:
        if not self._meets_min(event.severity):
            return
        payload = {
            "routing_key": self._routing_key,
            "event_action": "trigger",
            "dedup_key": f"latentspec:{event.agent_id}:{event.invariant_id}",
            "payload": {
                "summary": (
                    f"LatentSpec {event.severity.upper()} — "
                    f"{event.agent_name}: {event.invariant_description}"
                ),
                "source": event.agent_name,
                "severity": _SEV_MAP.get(event.severity, "warning"),
                "component": "latentspec",
                "group": event.agent_id,
                "class": event.outcome,
                "custom_details": {
                    "trace_id": event.trace_id,
                    "invariant_id": event.invariant_id,
                    "observed": event.observed,
                    "metadata": event.metadata,
                },
            },
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(self._endpoint, json=payload)
            resp.raise_for_status()
