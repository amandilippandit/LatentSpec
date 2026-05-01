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
  mutations in the per-type checker dispatchers — a real reviewer
  should re-run on Python 3.12.
- **Code review by a non-author.** `CODE_REVIEW.md` exists as a
  pointer, but no human other than me has read this code.
- **Real production agent traces.** I built shape-faithful simulators
  for AutoGPT, OpenDevin, BabyAGI from their published architecture
  docs — but I have no actual trace dumps from any deployed agent.

### Real-corpus pipeline surfaced limitations

From `scripts/realcorpus/run_realcorpus_pipeline.py`:

- **AutoGPT-style (200 traces, 200 distinct fingerprints)**: mining
  produces 62 invariants but ~30 are spurious chain rules
  (e.g. `parse_goal then observe then observe`) because PrefixSpan
  finds non-contiguous subsequences. On highly-variable agents this
  is mostly noise. The miner has no quality signal to drop them.
- **OpenDevin-style**: conditional invariants from the simulator's
  baked-in `"test" → run_tests` behaviour are NOT discovered by the
  statistical track. The MI association miner needs the keyword to
  appear in `min_keyword_traces` of traces — at small corpus sizes
  this floor is too high. Without LLM track active, real conditional
  patterns slip through.
- **BabyAGI-style** (the easy case): everything works as expected.
  Use this as the baseline of "the system functions correctly."

### Production scale unverified

- **No load test against deployed Postgres + Redis.** Integration
  tests use SQLite. Real Postgres-specific behaviour (lock contention,
  connection pool exhaustion under N parallel ingest workers, query
  planner choices on the TimescaleDB hypertable) is unmeasured.
- **No multi-process throughput measurement.** Benchmark runs in one
  process. Python's GIL means real production needs N workers, and
  cross-worker coordination through Redis pubsub is untested.
- **Z3 verification at 1+/sec sustained**: Z3 path tops out at ~357/sec
  but that's with no contention. Real deployment with concurrent
  certificate generation + streaming verification is unmeasured.

## What I would now trust the system to do

After external validation:

- ✅ Catch ordering / negative / state / composition / tool_selection
  / statistical violations on agents that match the booking-agent
  shape (5–30 tools, mostly recurring shapes, < 50 step traces).
- ✅ Survive arbitrary JSON inputs at the ingest endpoint without
  5xxing.
- ✅ Round-trip every model through SQL without crashing.
- ✅ Run streaming detection at 24K traces/sec single-process on a
  laptop.
- ✅ Verify a single Z3 invariant against a single trace without
  segfaulting.
- ✅ Reject malformed trace and params payloads with typed errors.

After external validation, I would **NOT** trust:

- ❌ Mining quality on agents like AutoGPT (very high fingerprint
  diversity, long traces, branching loops).
- ❌ Mining-recovered conditional rules without the LLM track.
- ❌ Z3 verification at >1/sec sustained.
- ❌ Multi-process production throughput claims.
- ❌ Anything below mutation-test kill-rate floor (untested on this
  Python version).

## Diff vs the prior "176 passing tests" claim

| Then | Now |
|---|---|
| 176 tests, all written by me, all passing | 210 tests + 4 benchmarks, including 30+ Hypothesis + 7 differential cases I did not hand-craft |
| "Streaming p99 < 100ms" claim, unmeasured | Measured: 59 μs (1700× headroom) |
| "Z3 verification works" claim, unmeasured | Discovered Z3 SIGSEGVs concurrently; fixed via dedicated executor |
| "Trace columns added" claim, untested | Discovered the model never had the columns; added them |
| "Closed-world negative miner" claim, untested by checker | Discovered the checker still rejected the new mode; rewrote it |
| Pipeline never run on non-self-built data | Three real-shape simulators run end-to-end with calibration + canonicalisation |

## What still needs external help

Things a non-author reviewer should look at first, ranked by risk:

1. The Z3 verifier lock (`src/latentspec/smt/verifier.py:_Z3_LOCK`) is
   per-process; if you run multiple Python processes (gunicorn workers)
   each gets its own Z3 — that's fine. But if you embed `latentspec`
   in a host process that has its own Z3 callers, the lock won't
   coordinate with them.
2. The `_route_to_cluster` function in `traces.py` does an O(n) scan
   over centroids per trace. At 1000+ centroids this becomes the
   ingest bottleneck. Use a vector DB query instead.
3. `register_or_update_version` is called on every ingest; the
   version-existence check is one SELECT per trace. Cache it per
   `(agent_id, version_tag)` in process memory.
4. The fingerprint distribution update happens synchronously in the
   ingest path; a 50-fingerprint diff is fine but a 500-fingerprint
   diff every trace is a real cost. Move to a background aggregator.
5. The PII redaction regex catalog is biased toward English / Latin
   character sets. Multi-language deploys need replacement patterns.

## Tools used

- **Hypothesis 6.152** — property-based test generation
- **mutmut 2.5 / 3.5** — both blocked on Python 3.14 (real tooling
  limit, not my code's fault)
- **aiosqlite + sqlalchemy 2.0** — async DB integration
- **FastAPI TestClient** — in-process HTTP fuzzing
- **z3-solver 4.13** — symbolic verification
- **scikit-learn 1.5** — clustering + isolation forest
- The benchmarks use only the standard library (`time.perf_counter_ns`,
  `statistics`).

## Reproducing this validation

```bash
.venv/bin/python -m pytest tests/                     # 210 unit + property + integration
.venv/bin/python benches/bench_streaming.py           # ~15s
.venv/bin/python benches/bench_z3_concurrency.py      # ~5s, MUST exit 0
.venv/bin/python scripts/realcorpus/run_realcorpus_pipeline.py  # ~10s
```

If `bench_z3_concurrency.py` exits non-zero, the dedicated executor
fix didn't apply correctly — that's a release blocker.
