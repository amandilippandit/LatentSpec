"""Slack webhook sink (§2.2 Alert Manager — Growth-phase priority)."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from latentspec.alerts.dispatcher import AlertEvent, AlertSink


_SEVERITY_EMOJI = {
    "critical": ":rotating_light:",
    "high":     ":warning:",
    "medium":   ":information_source:",
    "low":      ":small_blue_diamond:",
}


class SlackWebhookSink(AlertSink):
    name = "slack"

    def __init__(self, webhook_url: str, *, channel_override: str | None = None) -> None:
        self._url = webhook_url
        self._channel = channel_override

    async def send(self, event: AlertEvent) -> None:
        emoji = _SEVERITY_EMOJI.get(event.severity, ":warning:")
        when = datetime.fromtimestamp(event.detected_at, tz=UTC).isoformat()
        body = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji}  Behavioral regression — {event.severity.upper()}",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Agent*\n{event.agent_name}"},
                        {"type": "mrkdwn", "text": f"*Rule*\n`inv-{event.invariant_id[:8]}`"},
                        {"type": "mrkdwn", "text": f"*Trace*\n`{event.trace_id}`"},
                        {"type": "mrkdwn", "text": f"*Detected*\n{when}"},
                    ],
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"*Description*\n{event.invariant_description}\n\n"
                            f"*Observed*\n{event.observed or '_(no detail)_'}"
                        ),
                    },
                },
            ]
        }
        if self._channel:
            body["channel"] = self._channel
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(self._url, json=body)
            resp.raise_for_status()
