"""Z3 SMT backend (§3.2 / §10.1).

Compiles structured invariant params into Z3 expressions and verifies traces
against them. Two execution modes:

  - VERIFICATION mode (runtime): instantiate the trace as a concrete Z3 model
    and check whether the formula holds. Slower than the rule-based checkers
    but produces a counter-example when it doesn't.

  - PROOF mode (§10.1 enterprise): assert the formula over a constrained
    abstract trace and ask Z3 to prove it holds for ALL traces satisfying
    the constraints. The result is a signed certificate object — the
    artifact regulated-industry buyers pay for.

Today the verifier is the parallel diagnostic path. The fast rule-based
checkers stay on the hot path. Proof-mode certificates light up at month 6+
when enterprise demand pulls them forward, but the engine is in place now.
"""

from latentspec.smt.compiler import (
    Z3Compilation,
    Z3CompilerError,
    compile_invariant,
    compiler_for,
)
from latentspec.smt.certificates import VerificationCertificate, generate_certificate
from latentspec.smt.symbolic import (
    SymbolicAssumption,
    SymbolicProof,
    verify_symbolic,
)
from latentspec.smt.synthesis import synthesize_violating_trace
from latentspec.smt.verifier import VerificationResult, verify_trace

__all__ = [
    "SymbolicAssumption",
    "SymbolicProof",
    "VerificationCertificate",
    "VerificationResult",
    "Z3Compilation",
    "Z3CompilerError",
    "compile_invariant",
    "compiler_for",
    "generate_certificate",
    "synthesize_violating_trace",
    "verify_symbolic",
    "verify_trace",
]
