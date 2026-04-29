"""Tests for SDK PII redaction + sampling."""

from __future__ import annotations

from datetime import UTC, datetime

from latentspec.schemas.trace import (
    NormalizedTrace,
    ToolCallStep,
    TraceMetadata,
    UserInputStep,
)
from latentspec.sdk.redaction import NerBackend, NerEntity, Redactor
from latentspec.sdk.sampling import Sampler, SamplingStrategy


def _trace(*, trace_id: str = "trace-1", error: bool = False, latency_ms: int = 100) -> NormalizedTrace:
    return NormalizedTrace(
        trace_id=trace_id,
        agent_id="a",
        timestamp=datetime.now(UTC),
        steps=[
            UserInputStep(content="hello"),
            ToolCallStep(
                tool="t",
                args={},
                latency_ms=latency_ms,
                result_status="error" if error else "success",
            ),
        ],
        metadata=TraceMetadata(),
    )


def test_redactor_masks_email_and_credit_card_and_token() -> None:
    r = Redactor()
    # Token, email, and credit-card patterns are tested against synthetic
    # fixtures whose prefixes don't match real provider regexes (so GitHub
    # secret scanning doesn't false-positive on this file).
    out_email_cc = r.redact_string(
        "email me at jane@example.com about card 4242 4242 4242 4242"
    )
    assert "jane@example.com" not in out_email_cc
    assert "[redacted:email]" in out_email_cc
    assert "4242 4242 4242 4242" not in out_email_cc
    assert "[redacted:credit_card]" in out_email_cc

    # Use a synthetic prefix that the redactor's regex still matches but
    # provider-specific scanners don't.
    fake_token = "ls_" + "FAKEabcdef0123456789ABCDEF"
    out_token = r.redact_string(f"token: {fake_token}")
    assert "FAKEabcdef" not in out_token
    assert "[redacted:api_token]" in out_token


def test_redactor_blocks_sensitive_field_keys() -> None:
    r = Redactor()
    out = r.redact_value("password", "topsecret")
    assert out == "[redacted:blocked_field]"


def test_redactor_recurses_into_dict_and_list() -> None:
    r = Redactor()
    fake_token_value = "ls_" + "FAKEXXXXXXXXXXXXXXXXXXXXXXXXX"
    redacted = r.redact_value(
        None,
        {"user": "ann@example.com", "tokens": [fake_token_value]},
    )
    assert "ann@example.com" not in str(redacted)


def test_sampler_keeps_all_at_rate_one() -> None:
    s = Sampler(strategy=SamplingStrategy.RATE, rate=1.0)
    for i in range(20):
        assert s.keep(_trace(trace_id=f"t-{i}"))


def test_sampler_drops_all_at_rate_zero_keeping_errors() -> None:
    s = Sampler(strategy=SamplingStrategy.RATE, rate=0.0, keep_errors=True)
    assert not s.keep(_trace(trace_id="t-success"))
    assert s.keep(_trace(trace_id="t-err", error=True))

