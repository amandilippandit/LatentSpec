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


def test_sampler_deterministic_decision_stable_across_calls() -> None:
    s = Sampler(strategy=SamplingStrategy.RATE, rate=0.5, deterministic=True, keep_errors=False)
    decisions = [s.keep(_trace(trace_id="stable-trace-id")) for _ in range(5)]
    assert all(d == decisions[0] for d in decisions)


def test_adaptive_keeps_long_latency() -> None:
    s = Sampler(
        strategy=SamplingStrategy.ADAPTIVE,
        rate=0.0,
        long_latency_ms=500,
        keep_errors=False,
        keep_segment_rare=False,
    )
    assert not s.keep(_trace(trace_id="quick", latency_ms=100))
    assert s.keep(_trace(trace_id="slow", latency_ms=2000))


# ---- redaction extension --------------------------------------------------


def test_custom_redactor_pipeline_runs_after_patterns() -> None:
    """Custom redactors run on the post-pattern output."""
    r = Redactor()
    r.add_custom_redactor(
        lambda s: s.replace("PROJECT-ALPHA", "[redacted:internal_codename]")
    )
    out = r.redact_string(
        "Email user@example.com about PROJECT-ALPHA before launch"
    )
    assert "user@example.com" not in out
    assert "[redacted:email]" in out
    assert "PROJECT-ALPHA" not in out
    assert "[redacted:internal_codename]" in out


def test_redactor_handles_jwt_pattern() -> None:
    r = Redactor()
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    out = r.redact_string(f"Authorization: Bearer {jwt}")
    assert jwt not in out
    assert "[redacted:jwt]" in out


def test_ner_backend_redacts_detected_entities() -> None:
    """A pluggable NER backend's spans are masked alongside regex patterns."""
    text = "Jane Doe lives in Paris and works at OpenAI."

    class FakeNer(NerBackend):
        name = "fake"

        def detect(self, t: str) -> list[NerEntity]:
            spans: list[NerEntity] = []
            for needle, label in [("Jane Doe", "PER"), ("Paris", "LOC"), ("OpenAI", "ORG")]:
                idx = t.find(needle)
                if idx >= 0:
                    spans.append(NerEntity(label=label, start=idx, end=idx + len(needle)))
            return spans

    r = Redactor()
    r.add_ner_backend(FakeNer())
    out = r.redact_string(text)
    assert "Jane Doe" not in out
    assert "Paris" not in out
    assert "OpenAI" not in out
    assert "[redacted:per]" in out
    assert "[redacted:loc]" in out
    assert "[redacted:org]" in out


def test_ner_backend_failure_falls_back_silently() -> None:
    class CrashingNer(NerBackend):
        name = "crash"

        def detect(self, t: str) -> list[NerEntity]:
            raise RuntimeError("boom")

    r = Redactor()
    r.add_ner_backend(CrashingNer())
    # Pattern-based redaction must still run even when NER blows up.
    out = r.redact_string("email me at jane@example.com")
    assert "[redacted:email]" in out
