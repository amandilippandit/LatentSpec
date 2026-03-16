"""§4 regression check route — `POST /agents/{id}/check`.

Accepts:
  - inline baseline + candidate trace lists, OR
  - a `version_tag` pair (`baseline_version_tag`, `candidate_version_tag`)
    so callers can compare two saved trace populations against each other.

Returns the structured regression report plus a rendered §4.2 PR-comment.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from latentspec.checking.runner import invariant_to_spec
from latentspec.db import get_db
from latentspec.models import Agent, Invariant, InvariantStatus, Trace
from latentspec.regression.batch import compare_trace_sets, _exit_code_for
from latentspec.regression.report import format_pr_comment
from latentspec.schemas.trace import NormalizedTrace

router = APIRouter()


class CheckIn(BaseModel):
    baseline_traces: list[dict] | None = None
    candidate_traces: list[dict] | None = None
    baseline_version_tag: str | None = None
    candidate_version_tag: str | None = None
    fail_on: str = Field(default="critical")
    agent_name: str = Field(default="agent")
    limit: int = Field(default=500, ge=1, le=5000)


class CheckSummaryOut(BaseModel):
    invariant_id: str
    description: str
    severity: str
    type: str
    pass_rate: float
    fail_rate: float
    warn_rate: float
    sample_failure_traces: list[str]


class CheckOut(BaseModel):
    invariants_checked: int
    passes: int
    warnings: list[CheckSummaryOut]
    failures: list[CheckSummaryOut]
    exit_code: int
    report: str


def _summary_to_out(s) -> CheckSummaryOut:
    return CheckSummaryOut(
        invariant_id=s.invariant_id,
        description=s.description,
        severity=s.severity.value,
        type=s.type.value,
        pass_rate=s.pass_rate,
        fail_rate=s.fail_rate,
        warn_rate=s.warn_rate,
        sample_failure_traces=s.sample_failure_traces,
    )


@router.post("/{agent_id}/check", response_model=CheckOut)
async def run_check(
    agent_id: uuid.UUID,
