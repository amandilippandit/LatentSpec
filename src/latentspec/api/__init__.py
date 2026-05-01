"""FastAPI application for LatentSpec (§7).

Mounts:
  POST /traces                                 — ingest one trace
  POST /traces/batch                           — ingest many traces
  POST /agents                                 — register an agent
  GET  /agents                                 — list agents
  POST /agents/{id}/mining-runs                — trigger a mining run
  GET  /agents/{id}/invariants                 — list discovered invariants
  PATCH /invariants/{id}                       — confirm / reject / edit
"""
