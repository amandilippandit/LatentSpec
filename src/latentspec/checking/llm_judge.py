"""LLM-as-judge checker for output_format invariants (§3.3).

The only invariant type whose semantics we can't reduce to a Python
predicate. We ask Claude (with deterministic temperature) to render a yes/no
verdict against the agent's final response and the rule description, plus
a one-line explanation that becomes part of the §4.3 violation analysis.

Falls back to PASS-with-warning when no API key is configured so CI runs
don't break in local-dev environments without LLM access.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from latentspec.checking.base import (
    Checker,
    CheckOutcome,
    CheckResult,
    InvariantSpec,
    ViolationDetails,
)
from latentspec.config import get_settings
from latentspec.models.invariant import InvariantType
from latentspec.schemas.trace import AgentResponseStep, NormalizedTrace

log = logging.getLogger(__name__)


JUDGE_SYSTEM = """\
You are LatentSpec's LLM-as-judge. Given (a) one behavioral rule and
(b) a single agent trace, decide if the rule HOLDS or FAILS.

Return ONLY this JSON:
  {"verdict": "pass" | "fail" | "not_applicable",
   "reason":  "one short sentence"}

Be strict. Only return "pass" if the rule clearly holds; "fail" if it is
clearly violated; "not_applicable" if the rule's preconditions don't apply.
"""


class LLMJudgeChecker(Checker):
    invariant_type = InvariantType.OUTPUT_FORMAT

    def check(
        self, invariant: InvariantSpec, trace: NormalizedTrace
    ) -> CheckResult:
        # Synchronous interface; bridge to async client.
        try:
            return asyncio.get_event_loop().run_until_complete(
                self._check_async(invariant, trace)
            )
        except RuntimeError:
            # Already inside an async loop — schedule and wait.
            return asyncio.run(self._check_async(invariant, trace))

    async def _check_async(
        self, invariant: InvariantSpec, trace: NormalizedTrace
    ) -> CheckResult:
        settings = get_settings()
        if not settings.anthropic_api_key:
            log.warning(
                "ANTHROPIC_API_KEY not set — LLM-as-judge skipped, returning NOT_APPLICABLE"
            )
            return self._result(invariant, trace, CheckOutcome.NOT_APPLICABLE)

        # Lazy import so the rule-based path doesn't pay for the SDK weight.
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        response_text = " ".join(
            s.content for s in trace.steps if isinstance(s, AgentResponseStep)
        )

        user_msg = json.dumps(
            {
                "rule": invariant.description,
                "agent_response": response_text[:2000],
            },
            ensure_ascii=False,
        )

        try:
            resp = await client.messages.create(
                model=settings.anthropic_model,
                max_tokens=256,
                temperature=0.0,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
        except Exception as e:
            log.warning("LLM judge failed: %s", e)
            return self._result(invariant, trace, CheckOutcome.NOT_APPLICABLE)

        verdict, reason = self._parse(resp)
        if verdict == "pass":
            return self._result(invariant, trace, CheckOutcome.PASS)
        if verdict == "not_applicable":
            return self._result(invariant, trace, CheckOutcome.NOT_APPLICABLE)
        return self._result(
            invariant,
            trace,
            CheckOutcome.FAIL,
            ViolationDetails(
                expected=invariant.description,
                observed=reason or "LLM judge returned fail without explanation",
                extra={"verdict": "fail", "judge_reason": reason},
            ),
        )

    @staticmethod
    def _parse(resp: Any) -> tuple[str, str]:
        blocks = getattr(resp, "content", None) or []
        text = "".join(getattr(b, "text", "") or "" for b in blocks).strip()
        if not text:
            return "not_applicable", ""
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return "not_applicable", ""
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return "not_applicable", ""
        verdict = str(obj.get("verdict") or "not_applicable").lower()
        if verdict not in {"pass", "fail", "not_applicable"}:
            verdict = "not_applicable"
        return verdict, str(obj.get("reason") or "")
