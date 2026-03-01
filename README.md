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

