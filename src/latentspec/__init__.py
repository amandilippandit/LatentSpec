"""LatentSpec — discover behavioral invariants from AI agent traces.

Implements the technical plan §1–§10: a four-stage pipeline (ingest → mine →
detect → alert) that automatically discovers behavioral rules from production
agent traces, flags regressions on every PR, and **enforces critical rules
inline at runtime** via the guardrail decorator.

Public surface (§5.1 + production hardening):

    import latentspec
    latentspec.init(api_key="ls_...", agent_id="booking-agent")

    @latentspec.trace_tool
    def search_flights(dest: str, date: str): ...

    rules = latentspec.RuleSet.from_api(agent_id="booking-agent")

    @latentspec.guardrail(rules, fail_on="critical")
    def send_email(to: str, body: str): ...

    with latentspec.guarded_turn(rules, user_input="..."):
        search_flights(...)
        send_email(...)   # raises GuardrailViolation on critical breach
"""

from latentspec.sdk.client import (
    flush,
    get_client,
    init,
    is_initialized,
    record_trace,
    shutdown,
)
from latentspec.sdk.decorators import current_collector, trace, trace_tool
from latentspec.sdk.guardrail import (
    GuardrailContext,
    GuardrailViolation,
    RuleSet,
    guarded_turn,
    guardrail,
)
from latentspec.sdk import redaction, sampling

__version__ = "0.1.0"

__all__ = [
    "GuardrailContext",
    "GuardrailViolation",
    "RuleSet",
    "current_collector",
    "flush",
    "get_client",
    "guarded_turn",
    "guardrail",
    "init",
    "is_initialized",
    "record_trace",
    "redaction",
    "sampling",
    "shutdown",
    "trace",
    "trace_tool",
]
