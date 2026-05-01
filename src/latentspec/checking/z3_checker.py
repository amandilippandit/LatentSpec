"""Z3-backed checker — used for verification mode and certificate generation.

The default dispatch uses the type-specific rule-based checkers (sub-100ms).
This checker is the *parallel* path — slower (~10-50ms typical), but Z3
returns a counter-example object that's directly attachable to the §4.3
violation analysis. Useful for diagnostic deep-dives and as the engine
behind the §10.1 enterprise certificate generator.

Compiled formulae are cached by their stable signature so a second
invocation against the same invariant skips compilation.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from latentspec.checking.base import (
    Checker,
    CheckOutcome,
    CheckResult,
    InvariantSpec,
    ViolationDetails,
)
from latentspec.models.invariant import InvariantType
from latentspec.schemas.trace import NormalizedTrace
from latentspec.smt.compiler import Z3Compilation, compile_invariant
from latentspec.smt.verifier import verify_trace

log = logging.getLogger(__name__)


class Z3Checker(Checker):
    """Use the Z3 SMT verifier to check ANY invariant type for which a
    compiler exists. Slower than rule-based but produces counter-examples."""

    invariant_type = InvariantType.ORDERING  # placeholder; we override `accepts`

    def __init__(self, *, timeout_ms: int = 50) -> None:
        self._timeout_ms = timeout_ms
        self._cache: dict[str, Z3Compilation] = {}
        self._lock = threading.Lock()

    def accepts(self, invariant_type: InvariantType) -> bool:
        return invariant_type != InvariantType.OUTPUT_FORMAT

    def _compile_cached(self, invariant: InvariantSpec) -> Z3Compilation:
        key = f"{invariant.type.value}::" + ",".join(
            f"{k}={invariant.params.get(k)}" for k in sorted(invariant.params)
        )
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            compilation = compile_invariant(invariant.type, invariant.params)
            self._cache[key] = compilation
            return compilation

    def check(
        self, invariant: InvariantSpec, trace: NormalizedTrace
    ) -> CheckResult:
        try:
            compilation = self._compile_cached(invariant)
        except Exception as e:
            log.debug("Z3 compile failed for %s: %s", invariant.id, e)
            return self._result(invariant, trace, CheckOutcome.NOT_APPLICABLE)

        result = verify_trace(compilation, trace, timeout_ms=self._timeout_ms)
        if result.error:
            # Fail-open on Z3 errors (timeout / unknown) per §4.1 streaming SLA
            return self._result(invariant, trace, CheckOutcome.NOT_APPLICABLE)

        if result.holds:
            return self._result(invariant, trace, CheckOutcome.PASS)

        details = ViolationDetails(
            expected=invariant.description,
            observed="Z3 found a counter-example in this trace",
            extra={
                "z3_counter_example": result.counter_example or {},
                "z3_duration_ms": result.duration_ms,
            },
        )
        return self._result(invariant, trace, CheckOutcome.FAIL, details)
