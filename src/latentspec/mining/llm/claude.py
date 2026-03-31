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
            log.exception("Claude API call failed")
            raise LLMMiningError(f"Claude API call failed: {e}") from e

        text = self._extract_text(resp)
        return self._parse(text)

    @staticmethod
    def _extract_text(resp: Any) -> str:
        # Anthropic returns content blocks; concatenate text blocks.
        blocks = getattr(resp, "content", None) or []
        chunks: list[str] = []
        for block in blocks:
            t = getattr(block, "text", None)
            if t:
                chunks.append(t)
            elif isinstance(block, dict) and block.get("type") == "text":
                chunks.append(block.get("text", ""))
        return "".join(chunks).strip()

    @staticmethod
    def _parse(raw_text: str) -> list[InvariantCandidate]:
        if not raw_text:
            return []
        # Allow models that wrap JSON in ```json fences.
        cleaned = raw_text
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        # Heuristic: take from the first '{' to the last '}'.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1:
            log.warning("Claude response contained no JSON object")
            return []
        try:
            obj = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as e:
            log.warning("failed to parse Claude JSON: %s", e)
            return []

        items = obj.get("invariants") or []
        candidates: list[InvariantCandidate] = []
        for raw in items:
            try:
                inv_type = InvariantType(raw["type"])
                # Validate `params` against the per-type schema. Without this
                # the LLM can emit shapes the runtime checker can't read; we
                # drop those rather than persist unenforceable rules.
                raw_params = dict(raw.get("params") or {})
                try:
                    params = validate_params(inv_type, raw_params)
                except ParamsValidationError as e:
                    log.debug("LLM candidate failed params schema: %s", e)
                    continue

                candidates.append(
                    InvariantCandidate(
                        type=inv_type,
                        description=str(raw["description"]).strip(),
                        formal_rule=str(raw.get("formal_rule") or "").strip(),
                        evidence_trace_ids=[str(x) for x in raw.get("evidence_trace_ids", [])],
                        support=float(raw.get("support", 0.0)),
                        consistency=float(raw.get("consistency", 0.0)),
                        severity=Severity(raw.get("severity", "medium")),
                        discovered_by="llm",
                        extra=params,
                    )
                )
            except (KeyError, ValueError, ValidationError) as e:
                log.debug("skipping malformed candidate: %s", e)
                continue
        return candidates
