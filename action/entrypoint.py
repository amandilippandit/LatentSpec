#!/usr/bin/env python
"""GitHub Action entrypoint — wraps `latentspec check`.

Logic:
  1. Resolve the invariant set: prefer a local JSON file when provided,
     otherwise fetch the active set from the LatentSpec API.
  2. Run the regression comparison.
  3. Render the §4.2 PR-comment report and:
       - print to stdout (action log)
       - emit `report` and `exit-code` outputs
       - write to GITHUB_STEP_SUMMARY if available
  4. Exit with the severity-gated code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

import httpx

from latentspec.checking.base import InvariantSpec
from latentspec.models.invariant import InvariantType, Severity
from latentspec.regression.batch import _exit_code_for, compare_trace_sets
from latentspec.regression.report import format_pr_comment
from latentspec.schemas.trace import NormalizedTrace


def _load_traces(path: Path) -> list[NormalizedTrace]:
    raw = json.loads(path.read_text())
    return [NormalizedTrace.model_validate(item) for item in raw]


def _load_invariants_from_file(path: Path) -> list[InvariantSpec]:
    raw = json.loads(path.read_text())
    out: list[InvariantSpec] = []
    for item in raw:
        out.append(
            InvariantSpec(
                id=str(item.get("invariant_id") or item.get("id") or uuid.uuid4().hex),
                type=InvariantType(item["type"]),
                description=item["description"],
                formal_rule=item.get("formal_rule") or "",
                severity=Severity(item.get("severity", "medium")),
                params=item.get("params") or {},
            )
        )
    return out


def _fetch_invariants(api_base: str, api_key: str, agent_id: str) -> list[InvariantSpec]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    params = {"agent_id": agent_id, "status": "active"}
    resp = httpx.get(f"{api_base.rstrip('/')}/invariants", params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    rows = resp.json()
    out: list[InvariantSpec] = []
    for row in rows:
        out.append(
            InvariantSpec(
                id=str(row["id"]),
                type=InvariantType(row["type"]),
                description=row["description"],
                formal_rule=row.get("formal_rule") or "",
                severity=Severity(row.get("severity", "medium")),
                params=row.get("params") or {},
            )
        )
    return out


def _emit_outputs(report_text: str, exit_code: int) -> None:
    out_path = os.environ.get("GITHUB_OUTPUT")
    if out_path:
        delim = f"EOF_{uuid.uuid4().hex}"
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(f"report<<{delim}\n{report_text}\n{delim}\n")
            f.write(f"exit-code={exit_code}\n")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("```\n")
            f.write(report_text)
            f.write("\n```\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default="")
    parser.add_argument("--agent-id", default="")
    parser.add_argument("--agent-name", default="agent")
    parser.add_argument("--invariants", default="")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--fail-on", default="critical")
    parser.add_argument("--api-base", default="https://api.latentspec.dev")
    args = parser.parse_args()

    if args.invariants:
        invariants = _load_invariants_from_file(Path(args.invariants))
    elif args.agent_id:
        invariants = _fetch_invariants(args.api_base, args.api_key, args.agent_id)
    else:
        print("ERROR: either --invariants or --agent-id is required", file=sys.stderr)
        return 2

    baseline = _load_traces(Path(args.baseline))
    candidate = _load_traces(Path(args.candidate))

    report = compare_trace_sets(invariants, baseline, candidate)
    text = format_pr_comment(report, agent_name=args.agent_name)
    print(text)
    code = _exit_code_for(report, args.fail_on)
    _emit_outputs(text, code)
    return code


if __name__ == "__main__":
    sys.exit(main())
