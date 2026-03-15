"""Generic JSON webhook sink — fan-out to any HTTP endpoint."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import httpx

from latentspec.alerts.dispatcher import AlertEvent, AlertSink


class GenericWebhookSink(AlertSink):
    name = "webhook"

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        signing_secret: str | None = None,
    ) -> None:
        self._url = url
        self._headers = dict(headers or {})
        self._signing_secret = signing_secret

    async def send(self, event: AlertEvent) -> None:
        body: dict[str, Any] = asdict(event)
        headers = dict(self._headers)
        headers.setdefault("Content-Type", "application/json")
        if self._signing_secret:
            import hashlib
            import hmac
            import json

            payload = json.dumps(body, sort_keys=True).encode()
            sig = hmac.new(
                self._signing_secret.encode(), payload, hashlib.sha256
            ).hexdigest()
            headers["X-LatentSpec-Signature"] = f"sha256={sig}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(self._url, json=body, headers=headers)
            resp.raise_for_status()
