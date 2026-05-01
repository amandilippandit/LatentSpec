"""§10.1 formal verification certificate generator.

Two complementary modes:

  - **EMPIRICAL** (default): per-trace verification over a finite sample.
    Useful for "did this rule hold across the last N production traces?".
    What we shipped first.

  - **SYMBOLIC** (the §10.1 promise): bounded model checking via the
    symbolic verifier in `smt/symbolic.py`. Proves the rule holds for
    EVERY trace satisfying the bounded constraints — no sample, no
    statistics, just a Z3 proof. The output certificate carries the
    explicit assumptions (max trace length, latency bound, tool domain)
    so a regulator can independently re-verify by replaying the same
    formula through their own Z3.

Empirical mode complements rather than replaces symbolic mode:
  - Symbolic answers "is this rule mathematically guaranteed within bound L?"
  - Empirical answers "does this rule hold across observed production?"

Both are useful. Both ship as fields on the certificate.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from latentspec.schemas.trace import NormalizedTrace
from latentspec.smt.compiler import Z3Compilation
from latentspec.smt.symbolic import SymbolicAssumption, verify_symbolic
from latentspec.smt.verifier import verify_trace


CertificateMode = Literal["symbolic", "empirical", "combined"]


@dataclass
class SymbolicProofPayload:
    """The proof half of a combined certificate."""

    proven: bool
    duration_ms: float
    max_trace_length: int
    assumptions: list[dict[str, str]]
    counter_example: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class EmpiricalEvidencePayload:
    """The sample-attestation half of a combined certificate."""

    sample_size: int
    sample_holds: int
    sample_violates: int
    counter_examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class VerificationCertificate:
    certificate_id: str
    mode: CertificateMode
    invariant_signature: str
    invariant_description: str
    invariant_type: str
    issued_at: str
    issuer: str
    symbolic: SymbolicProofPayload | None = None
    empirical: EmpiricalEvidencePayload | None = None
    signature_hex: str | None = None


def _sign(payload: bytes, key_b: bytes | None) -> str | None:
    if not key_b:
        return None
    return hmac.new(key_b, payload, hashlib.sha256).hexdigest()


def _payload_for_signing(cert: VerificationCertificate) -> bytes:
    body = asdict(cert)
    body.pop("signature_hex", None)
    return json.dumps(body, sort_keys=True, default=str).encode()


def _empirical(
    compilation: Z3Compilation,
    sample: list[NormalizedTrace],
    *,
    timeout_ms_per_trace: int,
    max_counter_examples: int,
) -> EmpiricalEvidencePayload:
    holds = 0
    violates = 0
    counter_examples: list[dict[str, Any]] = []
    for trace in sample:
        result = verify_trace(compilation, trace, timeout_ms=timeout_ms_per_trace)
        if result.holds:
            holds += 1
        else:
            violates += 1
            if (
                result.counter_example is not None
                and len(counter_examples) < max_counter_examples
            ):
                counter_examples.append(
                    {"trace_id": trace.trace_id, "counter_example": result.counter_example}
                )
    return EmpiricalEvidencePayload(
        sample_size=len(sample),
        sample_holds=holds,
        sample_violates=violates,
        counter_examples=counter_examples,
    )


def _symbolic(
    compilation: Z3Compilation,
    *,
    max_trace_length: int,
    timeout_ms: int,
) -> SymbolicProofPayload:
    proof = verify_symbolic(
        compilation,
        max_length=max_trace_length,
        timeout_ms=timeout_ms,
    )
    return SymbolicProofPayload(
        proven=proof.proven,
        duration_ms=proof.duration_ms,
        max_trace_length=proof.max_trace_length,
        assumptions=[
            {"name": a.name, "value": a.value} for a in proof.assumptions
        ],
        counter_example=proof.counter_example,
        error=proof.error,
    )


def generate_certificate(
    compilation: Z3Compilation,
    sample: list[NormalizedTrace] | None = None,
    *,
    mode: CertificateMode = "combined",
    issuer: str = "latentspec-ce",
    timeout_ms_per_trace: int = 250,
    symbolic_max_trace_length: int = 12,
    symbolic_timeout_ms: int = 8000,
    signing_key_env: str = "LATENTSPEC_CERT_SIGNING_KEY",
    max_counter_examples: int = 3,
) -> VerificationCertificate:
    """Issue a §10.1 verification certificate.

    Args:
        compilation: the Z3 compilation of the invariant.
        sample: traces for empirical attestation (required when mode in
            {"empirical", "combined"}).
        mode: which proofs to include. Default `combined` runs both.
        signing_key_env: env var holding the HMAC signing key. Without it
            the certificate is unsigned.
    """
    cert = VerificationCertificate(
        certificate_id=f"cert-{uuid.uuid4().hex[:12]}",
        mode=mode,
        invariant_signature=compilation.signature,
        invariant_description=compilation.description,
        invariant_type=compilation.invariant_type.value,
        issued_at=datetime.now(UTC).isoformat(),
        issuer=issuer,
    )

    if mode in ("symbolic", "combined"):
        cert.symbolic = _symbolic(
            compilation,
            max_trace_length=symbolic_max_trace_length,
            timeout_ms=symbolic_timeout_ms,
        )
    if mode in ("empirical", "combined"):
        if sample is None:
            sample = []
        cert.empirical = _empirical(
            compilation,
            sample,
            timeout_ms_per_trace=timeout_ms_per_trace,
            max_counter_examples=max_counter_examples,
        )

    cert.signature_hex = _sign(
        _payload_for_signing(cert),
        os.environ.get(signing_key_env, "").encode() or None,
    )
    return cert


def verify_certificate_signature(
    cert: VerificationCertificate, *, signing_key: str
) -> bool:
    """Independent verification — re-compute HMAC and compare."""
    if cert.signature_hex is None:
        return False
    expected = _sign(_payload_for_signing(cert), signing_key.encode())
    if expected is None:
        return False
    return hmac.compare_digest(expected, cert.signature_hex)
