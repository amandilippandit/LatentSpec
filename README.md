<div align="center">

# LatentSpec

**Discover behavioral invariants from AI agent traces. Prove them with Z3. Block regressions inline.**

[![Tests](https://img.shields.io/badge/tests-210%20passing-brightgreen)](VALIDATION_REPORT.md)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Proprietary-lightgrey)](#license)

</div>

---

## What it is

LatentSpec is a behavioral guardrail layer for AI agents. It observes
production traces, automatically discovers the rules the agent
consistently follows, and enforces those rules — at PR time via a
GitHub Action, at runtime via an inline guardrail decorator, and as
formal Z3-backed verification certificates for regulated buyers.

The wedge: **52% of teams running AI agents in production run zero
offline evaluations.** Not because they don't care — because they
don't know what to test for. LatentSpec is a test suite that writes
itself.

## Why it exists

Three things break agents in production:

1. **Prompt change** — somebody tweaks a system prompt, behavior
   shifts silently, no test catches it.
2. **Model upgrade** — a new model version subtly changes tool-use
   patterns, no regression alarm fires.
3. **Tool drift** — a downstream API renames an endpoint, the agent
   adopts the new tool, an entire category of behavior quietly
   stops happening.

LatentSpec catches all three by treating *behavior* as a first-class
artifact: mine it, formalize it, check it on every change, alert
when it drifts.

## Architecture

```
┌──────────┐   ┌──────────────┐   ┌──────────────────────────┐   ┌────────────┐
│  INGEST  │──▶│  NORMALIZE   │──▶│        MINE (parallel)   │──▶│  TRIAGE    │
│  POST    │   │  LangChain / │   │  ┌────────────────────┐  │   │  >0.8 ▶ ✓  │
│  /traces │   │  raw JSON /  │   │  │ Track A statistical│  │   │  0.6-0.8 ⏸ │
│  SDK     │   │  DAG         │   │  │  PrefixSpan        │  │   │  <0.6   ✗ │
└──────────┘   └──────────────┘   │  │  distribution      │  │   └─────┬──────┘
                                  │  │  MI association    │  │         │
                                  │  │  closed-world neg  │  │         ▼
                                  │  │  isolation forest  │  │   ┌──────────┐
                                  │  └────────────────────┘  │   │FORMALIZE │
                                  │  ┌────────────────────┐  │   │ §3.2 obj │
                                  │  │ Track B  LLM       │  │   │ + Z3 SMT │
                                  │  │  Claude prompt     │  │   │ → DB     │
                                  │  └────────────────────┘  │   └─────┬────┘
                                  │           │              │         │
                                  │           ▼              │         ▼
                                  │  ┌────────────────────┐  │   ┌──────────┐
                                  │  │  cross-validate    │  │   │  CHECK   │
                                  │  │  TF-IDF cosine     │  │   │  per-type│
                                  │  └────────────────────┘  │   │  + judge │
                                  └──────────────────────────┘   └─────┬────┘
                                                                       ▼
                                                           ┌────────────────────┐
                                                           │ PR comment / SDK   │
                                                           │ guardrail / alerts │
                                                           │ Z3 certificates    │
                                                           └────────────────────┘
```

## Install

LatentSpec is not yet published to PyPI. Install from this repo:

### Option A — clone + editable install (recommended for most users)

```bash
git clone https://github.com/amandilippandit/latentspec.git
cd latentspec
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

That's enough to run the CLI, the demo, the test suite, and the
benchmarks — no database needed.

### Option B — direct install from GitHub (use as a library)

```bash
pip install "latentspec @ git+https://github.com/amandilippandit/latentspec.git"
```

Then `import latentspec` works in any virtualenv. You still need to
clone the repo if you want the migrations, the GitHub Action source,
or the dashboard.

### Requirements

- **Python 3.12+** (3.12, 3.13, or 3.14)
- **macOS or Linux** (Windows is untested)
- For the API server: **Docker** (or local Postgres 16 + Redis 7)
- For the dashboard: **Node.js 20+ + npm**

The pip install pulls in: FastAPI, SQLAlchemy 2.0, Anthropic SDK,
Z3, scikit-learn, Hypothesis, and ~25 other deps — first install
takes 2-3 minutes.

## Use it

There are five distinct ways to actually use LatentSpec. Pick the
one that matches your situation.

### 1. "I want to see what it does, no setup" (~30 seconds)

```bash
.venv/bin/latentspec demo --no-llm
```

Generates 240 synthetic agent traces, runs the full mining pipeline
in-memory, and prints the discovered behavioral rules to your
terminal. No DB, no API key, nothing else needed. You'll see ~30
invariants ranked by confidence.

### 2. "I have my own trace data" (~2 minutes)

Put your traces in a JSON file matching the §3.2 schema (one trace
per JSON object in a top-level array — `latentspec demo --out
traces.json` will dump an example).

```bash
# Mine your traces
.venv/bin/latentspec mine --traces my_traces.json --no-llm --out invariants.json

# Compare a candidate set against a baseline
.venv/bin/latentspec check \
  --invariants invariants.json \
  --baseline   baseline.json \
  --candidate  candidate.json \
  --fail-on    critical

# Export to Promptfoo
.venv/bin/latentspec export-promptfoo --invariants invariants.json --out promptfoo.yaml
```

To enable the LLM mining track, set `ANTHROPIC_API_KEY` in your
shell or in `.env` and drop the `--no-llm` flag.

### 3. "I want it as an inline guardrail in my Python agent"

Install via Option A or B above. Then in your code:

```python
import latentspec
from latentspec import RuleSet, guarded_turn, guardrail, GuardrailViolation

# One-time bootstrap
latentspec.init(api_key="ls_…", agent_id="booking-agent")

# Load active rules — from the API, or from a local JSON file
rules = RuleSet.from_local_file("booking-agent", "invariants.json")
# or: rules = RuleSet.from_api(agent_id="booking-agent")

@guardrail(rules, fail_on="critical", max_check_ms=80)
def confirm_recipient(email: str) -> None: ...

@guardrail(rules, fail_on="critical")
def send_email(to: str, body: str) -> None: ...

with guarded_turn(rules, user_input="email me the receipt"):
    confirm_recipient("user@example.com")
    send_email("user@example.com", "Your receipt is ready.")
    # raises GuardrailViolation if a critical rule would break
```

The `@trace_tool` decorator captures every call into a §3.2 trace
and ships it to your LatentSpec backend (or buffers when offline):

```python
@latentspec.trace_tool
def search_flights(dest: str, date: str):
    return flight_api.search(dest, date)
```

### 4. "I want to run the full backend — API + DB + dashboard"

```bash
# Start Postgres 16 + TimescaleDB + Redis
docker compose up -d

# Configure
cp .env.example .env
# (edit .env — at minimum set ANTHROPIC_API_KEY if you want LLM mining)

# Apply schema migrations
.venv/bin/alembic upgrade head

# Start the FastAPI server
.venv/bin/latentspec-api
# → http://localhost:8000  (OpenAPI: http://localhost:8000/docs)

# In a separate terminal: start the dashboard
cd dashboard
npm install
npm run dev
# → http://localhost:3000
```

Quick sanity check the API is alive:

```bash
curl http://localhost:8000/health
# {"status":"ok","version":"0.1.0"}
```

Register an agent + ingest a trace:

```bash
# Create an org + agent
curl -X POST http://localhost:8000/orgs \
  -H "Content-Type: application/json" \
  -d '{"name":"My Org","slug":"my-org"}'
# → {"id":"<org-uuid>", ...}

curl -X POST http://localhost:8000/agents \
  -H "Content-Type: application/json" \
  -d '{"org_id":"<org-uuid>","name":"booking-agent","framework":"langchain"}'
# → {"id":"<agent-uuid>", ...}

# Ingest a trace
curl -X POST http://localhost:8000/traces \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "<agent-uuid>",
    "format": "raw_json",
    "payload": {
      "steps": [
        {"type":"user_input","content":"book a flight"},
        {"type":"tool_call","tool":"search_flights","args":{"dest":"NRT"}},
        {"type":"agent_response","content":"found 3 flights"}
      ]
    }
  }'

# Trigger mining (returns a job_id; poll /jobs/{id} for status)
curl -X POST http://localhost:8000/agents/<agent-uuid>/jobs/mining \
  -H "Content-Type: application/json" \
  -d '{"config":{"limit":500}}'
```

### 5. "I want it as a CI gate on every PR"

Add this workflow to your repo:

```yaml
# .github/workflows/latentspec.yml
name: LatentSpec Behavioral Regression Check
on: [push, pull_request]
jobs:
  behavioral-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run LatentSpec regression check
        uses: docker://ghcr.io/amandilippandit/latentspec-action:latest
        with:
          api-key:    ${{ secrets.LATENTSPEC_API_KEY }}
          agent-id:   booking-agent
          baseline:   tests/fixtures/baseline_traces.json
          candidate:  tests/fixtures/candidate_traces.json
          fail-on:    critical
```

> Note: the published `latentspec/action@v1` Marketplace listing is on
> the roadmap. For now you can either:
> 1. Build the image yourself: `docker build -t latentspec-action -f action/Dockerfile .` and reference it via `uses: ./action`
> 2. Or invoke the CLI directly in a `run:` step after `pip install`.

---

## Verify your install

```bash
# 1) Run the test suite
.venv/bin/python -m pytest -q
# → 210 passed in ~30s

# 2) Run the synthetic demo
.venv/bin/latentspec demo --no-llm
# → discovers ~30 invariants, prints them ranked by confidence

# 3) Run the benchmarks (real measured numbers on your machine)
.venv/bin/python benches/bench_streaming.py
.venv/bin/python benches/bench_z3_concurrency.py   # MUST exit 0

# 4) Run the 3-corpus harness
.venv/bin/python scripts/realcorpus/run_realcorpus_pipeline.py
```

If any of these breaks, that's a real bug — file an issue.

---

## Quickstart (TL;DR — for if you skipped above)

```bash
git clone https://github.com/amandilippandit/latentspec.git && cd latentspec
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# 1) Run the synthetic booking-agent demo end-to-end
.venv/bin/latentspec demo --no-llm

# 2) Mine a trace file → produce an invariant set
.venv/bin/latentspec mine --traces traces.json --no-llm --out invariants.json

# 3) Check a candidate trace set against the invariant set
.venv/bin/latentspec check \
  --invariants invariants.json \
  --baseline   baseline.json \
  --candidate  candidate.json \
  --fail-on    critical

# 4) Export to Promptfoo for the existing eval-tool ecosystem
.venv/bin/latentspec export-promptfoo --invariants invariants.json --out promptfoo.yaml
```

Run the test suite:

```bash
.venv/bin/python -m pytest -q                       # 210 unit + property + differential + integration
.venv/bin/python benches/bench_streaming.py         # latency + throughput
.venv/bin/python benches/bench_z3_concurrency.py    # Z3 SIGSEGV regression check
.venv/bin/python scripts/realcorpus/run_realcorpus_pipeline.py
```

## Inline guardrail (production runtime)

```python
import latentspec
from latentspec import RuleSet, guarded_turn, guardrail, GuardrailViolation

rules = RuleSet.from_api(agent_id="booking-agent")

@guardrail(rules, fail_on="critical", max_check_ms=80)
def confirm_recipient(email: str) -> None: ...

@guardrail(rules, fail_on="critical")
def send_email(to: str, body: str) -> None: ...

with guarded_turn(rules, user_input="email me the receipt"):
    confirm_recipient("user@example.com")
    send_email("user@example.com", "...")           # passes — precondition met

with guarded_turn(rules, user_input="email me"):
    try:
        send_email("user@example.com", "...")       # raises GuardrailViolation
    except GuardrailViolation as v:
        log.warning("blocked: %s (severity=%s)", v.invariant_description, v.severity)
```

## Z3 verification

Three modes, real bounded model checking:

```python
from latentspec.smt import (
    compile_invariant,
    verify_trace,                  # 1. concrete per-trace verification
    verify_symbolic,               # 2. bounded model-checking proof
    synthesize_violating_trace,    # 3. adversarial trace synthesis
    generate_certificate,          #    + signed combined certificates
)
from latentspec.models.invariant import InvariantType

comp   = compile_invariant(InvariantType.ORDERING, {"tool_a": "auth", "tool_b": "db_write"})
result = verify_trace(comp, my_trace, timeout_ms=100)
proof  = verify_symbolic(comp, max_length=12, timeout_ms=8000)
adv    = synthesize_violating_trace(comp, max_length=6)
cert   = generate_certificate(comp, sample=recent_traces, mode="combined")
```

The symbolic verifier quantifies over an abstract trace of length
`n ∈ [0, max_length]` and asks Z3 whether `Not(formula)` is
satisfiable. `unsat` ⇒ the rule is mathematically guaranteed within
the bound.

The adversarial synthesizer is the inverse: ask Z3 for a model where
the formula is violated, decode the model into a `NormalizedTrace`,
then feed it through the runtime guardrail to confirm the guardrail
actually blocks it. Closes the prove-the-guardrail-actually-blocks
loop end-to-end.

## What's underneath each promise

| Promise | Algorithm |
|---|---|
| Discover ordering rules | PrefixSpan with closed-pattern pruning, projected-database recursion, gap-aware directional consistency |
| Discover conditional rules | Mutual information + Pearson chi-square (df=1, p<0.01) + lift floor over 2x2 contingency tables |
| Discover negative rules | Closed-world repertoire mining + customer denylist policies |
| Discover anomaly envelopes | IsolationForest over an 8-feature behavioral vector |
| Discover semantic rules | Anthropic Claude with schema-enforced output, batched 50–100 traces |
| Cross-validate two tracks | TF-IDF + character-trigram cosine clustering with params-conflict gating |
| Per-agent threshold tuning | Kneedle elbow detection + BH-FDR + chi-square critical lookup |
| Tool-name canonicalisation | 3-pass: normalised exact match → token Jaccard + Levenshtein → trigram cosine |
| Prove rules formally | Bounded model checking via Z3 over an uninterpreted trace model |
| Generate adversarial cases | Z3 model synthesis decoded into NormalizedTrace |
| Block at runtime | `@latentspec.guardrail` raises `GuardrailViolation` before the wrapped tool runs |
| Detect online drift | Page-Hinkley sequential test + CUSUM, per-`(agent, invariant)` |
| Issue certificates | HMAC-signed symbolic + empirical artifacts |
| Redact PII | 9 regex patterns + field blocklist + pluggable NER backends + custom redactor pipeline |

## Stack

- **Backend**: Python 3.12+, FastAPI, SQLAlchemy 2.0 async
- **Storage**: PostgreSQL 16 + TimescaleDB hypertables + pgvector
- **Cache + pubsub**: Redis
- **LLM**: Anthropic Claude (provider-shaped — swappable)
- **SMT**: Z3 (concrete + symbolic + adversarial synthesis)
- **ML**: scikit-learn (IsolationForest, k-means with silhouette)
- **Frontend**: Next.js 14 + Tailwind + Recharts
- **CI**: GitHub Action wrapping the same `latentspec check` CLI used locally

## Layout

```
src/latentspec/
├── active_learning/    Synthetic trace generator + persisted HITL queue
├── alerts/             Slack / webhook / PagerDuty dispatcher
├── api/                FastAPI app + middleware + 40 routes
├── auth/               API-key generation + hashing
├── calibration/        Per-agent threshold learning
├── canonicalization/   3-pass tool-name alias detection
├── checking/           7 rule-based checkers + Z3Checker + LLM judge
├── cli/                latentspec command (mine / demo / check / export-promptfoo)
├── exporters/          Promptfoo YAML + Guardrails AI / pytest stubs
├── jobs/               Background job runner (swappable for Celery)
├── mining/
│   ├── statistical/    PrefixSpan + distribution + MI association
│   │                   + closed-world negative + isolation-forest anomaly
│   ├── llm/            Anthropic Claude with structured prompt
│   ├── embeddings.py   TF-IDF + char-trigram cosine merge
│   ├── fingerprint.py  Trace shape hash + chi-square drift
│   ├── clustering.py   Multi-view vectorizer + k-means with silhouette
│   ├── cluster_orchestrator.py
│   └── subgraph.py     Frequent edges + paths + forks over DAG traces
├── models/             ORM (5 product + 3 auth + 9 extension tables)
├── normalizers/        LangChain + raw JSON → §3.2 schema
├── observability/      structlog + Prometheus metrics
├── packs/              Vertical packs: ecommerce / banking / healthcare
├── regression/         Batch comparison + PR comment + root-cause
├── schemas/            Pydantic wire schemas (trace, invariant, params, DAG)
├── sdk/                @guardrail + @trace_tool + redaction + sampling
├── sessions/           Multi-turn session model + orchestrator
├── smt/                Compiler + verifier + symbolic + synthesis + certificates
├── streaming/          Hot invariant cache + sub-100ms detector + drift
└── versioning/         AgentVersion tracker + tool-repertoire diff

action/                 GitHub Action (action.yml + Dockerfile + entrypoint)
alembic/                Migrations: 0001 initial · 0002 auth · 0003 extensions
benches/                Performance + Z3-concurrency benchmarks
dashboard/              Next.js dashboard (Agent Overview / Invariant Explorer / Trace Inspector)
scripts/                Demo agent + 3-corpus harness (AutoGPT / OpenDevin / BabyAGI shapes)
tests/                  210 tests across 30+ modules
  ├── property/         Hypothesis property-based tests
  ├── differential/     Production vs reference checker cross-validation
  ├── integration/      Real-DB round-trips via aiosqlite
  └── (root)            Unit tests for every module
```

## Validation

This is the part most agent-tooling repos skip:

- **210 tests** across unit, property-based (Hypothesis), differential
  (production vs reference impl), and integration (real-DB round-trip)
- **4 benchmark scripts** with measured numbers (not claimed):
  - p99 streaming latency: **59 µs** (24 invariants/agent)
  - sustained throughput: **24,395 traces/sec**
  - Z3 verification: **357/sec** with the dedicated executor
- **3 real-shape agent simulators** (AutoGPT, OpenDevin, BabyAGI) end-to-end
- **API schema fuzzing**: 330+ random JSON payloads, zero 5xx errors
- **External-reviewer checklist** in [CODE_REVIEW.md](CODE_REVIEW.md)

The validation pass surfaced and fixed three real bugs the prior
closed-loop test suite had missed — see [VALIDATION_REPORT.md](VALIDATION_REPORT.md)
for the honest accounting.

## Configuration

Every knob in `.env` (see `.env.example`). Highlights:

| Setting | Default | What it controls |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://…` | Async DB URL (FastAPI) |
| `REDIS_URL` | `redis://localhost:6379/0` | Cache + pubsub channel |
| `ANTHROPIC_API_KEY` | *(empty)* | Track B mining + LLM-as-judge + root cause |
| `MINING_BATCH_SIZE` | `75` | Track B trace batch size |
| `CONFIDENCE_REJECT_THRESHOLD` | `0.6` | Below ⇒ auto-reject (overridden by per-agent calibration) |
| `CONFIDENCE_REVIEW_THRESHOLD` | `0.8` | Above ⇒ auto-activate (overridden by per-agent calibration) |
| `LATENTSPEC_REQUIRE_API_KEY` | unset | Enforce API-key middleware on every request |
| `LATENTSPEC_CERT_SIGNING_KEY` | unset | HMAC key for signed verification certificates |

## Running with the full stack

```bash
docker compose up -d                     # Postgres + TimescaleDB + Redis
.venv/bin/alembic upgrade head           # Apply 0001 + 0002 + 0003
.venv/bin/latentspec-api                 # FastAPI on :8000

# Separate terminal:
cd dashboard && npm install && npm run dev   # http://localhost:3000
```

## CI integration

```yaml
# .github/workflows/latentspec.yml
name: LatentSpec Behavioral Regression Check
on: [push, pull_request]
jobs:
  behavioral-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: latentspec/action@v1
        with:
          api-key:    ${{ secrets.LATENTSPEC_API_KEY }}
          agent-id:   booking-agent
          baseline:   tests/fixtures/baseline_traces.json
          candidate:  tests/fixtures/candidate_traces.json
          fail-on:    critical
```

## Status

Production-shaped, not production-deployed. The pipeline is complete:
ingest → mine → check → block → alert. The validation pass surfaced
real limits documented in `VALIDATION_REPORT.md`. What's still ahead
is real-corpus testing against deployed agents, multi-process
throughput measurement, and external code review.

## License

Proprietary — all rights reserved.

---

<div align="center">
<sub>Built for AI agents that need to behave the way you actually meant.</sub>
</div>
