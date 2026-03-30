"""Anthropic Claude client for Track B mining.

Uses the Anthropic SDK to call Claude with the §3.2 structured-extraction
prompt. The response is required to be valid JSON; parse + validate to a
list of `InvariantCandidate`.

The class is provider-shaped so a future swap to a different LLM only
requires a sibling file (e.g. `openai.py`) implementing the same interface.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from anthropic import AsyncAnthropic
from pydantic import ValidationError

from latentspec.config import get_settings
from latentspec.mining.llm.prompts import SYSTEM_PROMPT, build_user_message
from latentspec.models.invariant import InvariantType, Severity
from latentspec.schemas.invariant import InvariantCandidate
from latentspec.schemas.params import ParamsValidationError, validate_params
from latentspec.schemas.trace import NormalizedTrace

log = logging.getLogger(__name__)


class LLMMiningError(RuntimeError):
    pass


class ClaudeMiner:
    """Async Track B miner backed by the Anthropic API."""

    def __init__(
        self,
        *,
        client: AsyncAnthropic | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> None:
        settings = get_settings()
        if client is None:
            if not settings.anthropic_api_key:
                raise LLMMiningError(
                    "ANTHROPIC_API_KEY is not set; cannot run LLM mining track"
                )
            client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._client = client
        self._model = model or settings.anthropic_model
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def mine_batch(
        self, traces: list[NormalizedTrace]
    ) -> list[InvariantCandidate]:
        """Run one extraction call against a batch of 50–100 traces."""
        if not traces:
            return []

        user_message = build_user_message(traces)
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as e:  # network / API errors
