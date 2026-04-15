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
