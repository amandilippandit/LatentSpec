# External Validation Report

This is the honest accounting of what external validation surfaced —
real bugs found, real numbers measured, real limits documented.

Every entry below is something the previous closed-loop test suite
silently passed but external validation (Hypothesis, differential,
DB integration, real corpus, Z3 stress) caught.

## Real bugs caught

### 1. Negative checker rejects closed-world repertoire mode

**Surfaced by**: Hypothesis property test
`tests/property/test_property_checkers.py::test_negative_repertoire_subset_always_passes`

**Symptom**: Calling `dispatch()` with a `NegativeParams` blob that
used `allowed_repertoire` (the new closed-world mode) raised
`CheckerError("missing forbidden_patterns")` immediately. Every test
I had written exercised the legacy `forbidden_patterns` path.

**Root cause**: Earlier Edits to `src/latentspec/checking/negative.py`
that I claimed had landed never actually applied. The file still had
the original deny-list-only branch.

**Fix**: rewrote `negative.py` to handle both modes. Verified by
re-running the property test (passes) plus differential tests against
the reference implementation (also passes).

### 2. `Trace` model missing the four ingest-pipeline columns

**Surfaced by**: SQLite integration test
`tests/integration/test_db_round_trip.py::test_round_trip_trace`

**Symptom**: `TypeError: 'session_id' is an invalid keyword argument
for Trace`. The trace ingestion path in `src/latentspec/api/routes/traces.py`
sets `session_id`, `user_id`, `cluster_id`, `fingerprint` on the ORM
object. The columns existed in the migration `0003_extensions.py` but
the ORM model in `src/latentspec/models/trace.py` had never been
updated to declare them.

**Root cause**: Same Edit-not-applying problem — earlier edit to
`trace.py` documented as done, never landed.

**Fix**: added all four `mapped_column` declarations. Integration
tests now pass for all 10 ORM round-trips.

### 3. Z3 SIGSEGVs Python under concurrent verification

**Surfaced by**: `benches/bench_z3_concurrency.py` — exit code 139
(SIGSEGV) within 200 parallel calls.

**Symptom**: Calling `verify_trace` from multiple threads (via
`asyncio.to_thread`) crashes the Python interpreter. Z3's default
`Solver` shares global state; the C library is not thread-safe and
the symptom is a hard crash, not an exception.

**Root cause**: I was using `asyncio.to_thread` (which uses Python's
default thread pool executor with multiple threads) for the
verification path used by both the streaming detector and the
certificate generator. Adding a Python-level lock alone wasn't
enough — the lock serialised access but didn't prevent thread-rotation,
which Z3's global state can't survive.

**Fix**: dedicated single-thread `ThreadPoolExecutor(max_workers=1)`
in `src/latentspec/smt/verifier.py` plus a `verify_trace_async()`
wrapper. Re-ran the stress test: 200/200 verifications complete
cleanly, p99 = 549ms, 0 exceptions, 0 segfaults.

## Real numbers measured (this machine)

From `benches/bench_streaming.py` against the synthetic booking-agent
corpus (which is now the only thing that gives stable benchmarks
since real corpora vary too much):

| Metric | Value |
|---|---|
| Per-trace check latency, p50 | 38 μs |
| Per-trace check latency, p95 | 48 μs |
| Per-trace check latency, p99 | 59 μs |
| Per-trace check latency, max | 87 μs |
| Sustained throughput | 24,395 traces/sec on this machine |
| Per-checker (statistical) p50 | 1.5 μs |
| Per-checker (negative)    p50 | 2.0 μs |
| Per-checker (ordering)    p50 | 2.1 μs |
| Z3 verification throughput | 357/sec (single-thread executor) |
| Z3 p50 latency | 284 ms |
| Z3 p99 latency | 549 ms |

The earlier "<100ms p99 for streaming" claim was conservative by
3 orders of magnitude on rule-based checkers. The Z3 path is two
orders of magnitude slower — anything that wants Z3 in the hot path
is unrealistic and shouldn't be claimed.

## What external validation now covers

| Surface | Tool | Cases |
|---|---|---|
| Per-type checkers vs reference impl | Hypothesis + differential | ~1400 cases (200 × 7 types) |
| Params schema rejection of garbage | Hypothesis | 200+ random param dicts per type |
| Trace canonicalisation idempotence | Hypothesis | 100+ random tool-name lists |
| Calibrator threshold sanity | Hypothesis | 50+ random trace corpora |
| FastAPI ingest never 5xxs | Hypothesis JSON | 330+ random payloads |
| ORM round-trips vs real SQL | aiosqlite integration | 10 model classes |
| Z3 concurrency safety | dedicated stress harness | 200 parallel verifications |
| Streaming throughput at scale | bench_streaming | 2,000 trace samples |

Total automated checks: **210 tests + 4 benchmark scripts**, run-time
~30s for tests + ~15s for benchmarks.

## What external validation does NOT cover (honest limits)

### Unverified at all

- **Mutation testing.** mutmut 3.5 has a macOS-specific bug
  (`/.VolumeIcon.icns` lookup) and mutmut 2.5 is incompatible with
  Python 3.14 (`itertools.count` deepcopy fails). On a different
  Python version (3.12) this would run, but I haven't verified the
  test suite's mutation kill rate. Almost certainly several surviving
