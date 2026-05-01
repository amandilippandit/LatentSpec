"""Normalizer for already-§3.2-shaped payloads.

This is the universal fallback (§5.2 P3 "Custom REST API — Already built").
It validates that the payload conforms to the §3.2 schema and accepts it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from latentspec.normalizers.base import Normalizer, NormalizerError, registry
from latentspec.schemas.trace import NormalizedTrace


class RawJSONNormalizer(Normalizer):
    name = "raw_json"

    def normalize(self, payload: dict[str, Any], *, agent_id: str) -> NormalizedTrace:
        # Accept the §3.2 shape directly. Fill in trace_id / timestamp if absent.
        payload = {**payload}
        payload.setdefault("trace_id", str(uuid.uuid4()))
        payload.setdefault("agent_id", agent_id)
        payload.setdefault("timestamp", datetime.now(UTC).isoformat())
        if "steps" not in payload:
            raise NormalizerError("raw_json payload missing 'steps'")

        try:
            return NormalizedTrace.model_validate(payload)
        except Exception as e:
            raise NormalizerError(f"invalid raw_json trace: {e}") from e


registry.register(RawJSONNormalizer())
