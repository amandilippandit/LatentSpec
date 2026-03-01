# External code review checklist

This document is the entry point for someone reviewing this codebase
who didn't write it. It exists because every test in this repo was
written by the same person who wrote the implementation — closed-loop
validation. Real validation requires an outside read.

A reviewer should be able to answer **yes** to each question below
after a focused pass. If they answer **no** anywhere, that's a real
gap to fix before relying on the system.

---

## 1. Mining → invariant pipeline

- Does the closed-world repertoire miner emit a `NegativeParams` blob
  with **`allowed_repertoire`** populated (not `forbidden_patterns`)?
  See `src/latentspec/mining/statistical/negative.py`.
- Are mining-produced ordering candidates **per-pair distinct**? If
  the same `(tool_a, tool_b)` appears with two different supports, the
  cross-validation merge upstream is broken.
- Is `formalize()` called on **every** candidate? The orchestrator
  drops candidates whose `extra` fails `validate_params`; confirm the
  drop count is logged and bounded (< 5% of candidates).

## 2. Per-type checkers

- For each of the 7 rule-based checkers (excluding LLM judge), open
  `src/latentspec/checking/<type>.py` and verify:
  - The check function never raises an unhandled exception. The only
    exception that should escape is `CheckerError`, which the runner
    catches and converts to `NOT_APPLICABLE`.
  - `NOT_APPLICABLE` is returned when the rule's preconditions are
    absent (e.g. negative checker has `forbidden_patterns` empty AND
    `allowed_repertoire` empty).
  - The reference implementation in `tests/differential/reference.py`
    matches the checker's semantics. Disagreement = bug in one of them.

## 3. Streaming detector

- The `StreamingDetector.check()` method MUST guarantee its return
  within `check_budget_ms`. Confirm the deadline arithmetic is correct
  (see `src/latentspec/streaming/detector.py`).
- Verify the detector calls `cache.get_local()` first, falling back to
  `cache.get_or_load()` only on miss.
- Drift events are populated **only when** the result outcome is PASS
  or FAIL (NOT_APPLICABLE results don't update PH/CUSUM state).

## 4. Z3 verification

- **Critical**: `verify_trace` must hold `_Z3_LOCK` for the entire
  solver lifecycle. Z3 is NOT thread-safe and SIGSEGVs Python under
  concurrent access (verified by `benches/bench_z3_concurrency.py`).
- All async callers MUST go through `verify_trace_async()`, never
  `asyncio.to_thread(verify_trace, ...)`. The async wrapper pins Z3
  to a single dedicated OS thread.
- The `max_length` bound on `verify_symbolic` is the proof scope:
  `proven=True` only means "for all traces of length ≤ L". Real
  traces routinely exceed L=12. Document the bound on every
  certificate.

## 5. Tool-name canonicalisation

- `canonical_form()` strips trailing version suffixes (`_v1`,
  `.v2.1`, `-3`) only — internal tokens like `payments_v2_execute`
  should NOT be re-stripped midstream.
- The three-pass alias detector (exact → token-Jaccard + edit
  distance → trigram cosine) must be **idempotent**: running
  `canonicalise()` twice on the same name yields the same result.
- Pick-canonical heuristic: shortest normalised form wins, ties broken
  lexicographically. Sanity: `["payments_v1_legacy", "payments_v2"]`
  should canonicalise to `payments_v2`.

## 6. Per-agent calibration

- Calibration **never** returns thresholds outside their physical
  range (`mining_min_support ∈ [0.3, 0.85]`, `confidence_review ∈
  [0.6, 0.99]`, etc.). Property test:
  `tests/property/test_property_calibration.py`.
- The elbow detection on the support curve falls back gracefully when
  the curve is flat (no candidates from permissive mining).
- For tiny corpora (< 30 traces), calibration **MUST** return defaults
  rather than producing unstable estimates.
