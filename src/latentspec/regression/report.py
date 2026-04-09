"""§4.2 PR-comment renderer + a richer terminal view.

PR-comment format (deliberately matches the doc):

    LatentSpec Behavioral Regression Report
    ========================================
    Agent: <name> | Invariants checked: 47
    PASS (44) | WARN (1) | FAIL (2)

    FAILURES:
      [CRITICAL] inv-0042: Agent no longer calls check_inventory before create_order
                 (violated in 23% of test traces)
      [HIGH]     inv-0019: Email confirmation step skipped when user language is Japanese
                 (0% compliance vs 100% baseline)

    WARNINGS:
      [MEDIUM]   inv-0031: Average response latency increased from 340ms to 890ms
                 for complex queries
"""

from __future__ import annotations

from typing import Mapping

from latentspec.models.invariant import Severity
from latentspec.regression.batch import (
    InvariantBatchSummary,
    RegressionReport,
)


def _short_id(invariant_id: str) -> str:
    """Return a stable 8-char tag suitable for `inv-XXXX` PR labels."""
    s = invariant_id
    if s.startswith("inv-"):
        s = s[4:]
    return s[:8]


def _baseline_lookup(report: RegressionReport) -> dict[str, InvariantBatchSummary]:
    return {bl.invariant_id: bl for bl in report.baseline}


def _format_failure(
    summary: InvariantBatchSummary,
    baseline: InvariantBatchSummary | None,
    *,
    severity_tag: str,
) -> str:
    fail_pct = round(summary.fail_rate * 100)
    if baseline is None:
        comparison = f"violated in {fail_pct}% of test traces"
    else:
        bl_pass = round(baseline.pass_rate * 100)
        cd_pass = round(summary.pass_rate * 100)
        comparison = f"{cd_pass}% compliance vs {bl_pass}% baseline"
    head = f"  [{severity_tag:<8}] inv-{_short_id(summary.invariant_id)}: {summary.description}"
    body = f"             ({comparison})"
    return f"{head}\n{body}"


def _format_warning(
    summary: InvariantBatchSummary,
    baseline: InvariantBatchSummary | None,
) -> str:
    if baseline is not None:
        bl_warn = round(baseline.warn_rate * 100)
        cd_warn = round(summary.warn_rate * 100)
        comparison = f"warning rate {bl_warn}% -> {cd_warn}%"
    else:
        comparison = f"warned on {round(summary.warn_rate * 100)}% of test traces"
