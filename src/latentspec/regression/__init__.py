"""Behavioral regression detection (§4.1 batch + §4.2 CI/CD + §4.3 violation analysis).

Three pieces:
  - batch:      compare a baseline trace set against a candidate trace set
  - report:     render the §4.2 PR-comment / CLI summary
  - root_cause: §4.3 LLM-generated diff explanations between baseline & candidate
"""

from latentspec.regression.batch import (
    InvariantBatchSummary,
    RegressionReport,
    compare_trace_sets,
)
from latentspec.regression.report import format_pr_comment, format_terminal
from latentspec.regression.root_cause import generate_root_cause_hints

__all__ = [
    "InvariantBatchSummary",
    "RegressionReport",
    "compare_trace_sets",
    "format_pr_comment",
    "format_terminal",
    "generate_root_cause_hints",
]
