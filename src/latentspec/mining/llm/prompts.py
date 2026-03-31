"""Structured extraction prompt for Track B (§3.2).

The prompt asks Claude to identify behavioral rules the agent consistently
follows — the same question a human reviewer would ask after reading a stack
of traces. Response is constrained to JSON matching the §3.3 taxonomy.
"""

from __future__ import annotations

import json
from typing import Any

from latentspec.schemas.trace import NormalizedTrace


SYSTEM_PROMPT = """\
You are LatentSpec's behavioral-invariant analyst. You read batches of
agent traces and identify behavioral rules the agent consistently follows
in production — the implicit specifications nobody documented.

You return ONLY a JSON object. No prose, no markdown, no commentary.

Output schema:
{
  "invariants": [
    {
      "type": "ordering" | "conditional" | "negative" | "statistical" |
              "output_format" | "tool_selection" | "state" | "composition",
      "description": string,
      "formal_rule": string,
      "evidence_trace_ids": string[],
      "support": number,
      "consistency": number,
      "severity": "low" | "medium" | "high" | "critical",
      "params": object
    }
  ]
}

The `params` field is REQUIRED and its shape is type-specific. The runtime
checker reads `params` directly — malformed shapes are silently discarded.
Use exactly these shapes:

  ordering:       {"tool_a": "<tool>", "tool_b": "<tool>"}
                  optionally: "chain": ["a","b","c"] for length 3+ chains.

  conditional:    {"keyword": "<lowercase keyword>", "tool": "<tool>"}

  negative:       {"forbidden_patterns": ["<tool>", ...], "category": "<label>"}
                  OR closed-world: {"allowed_repertoire": ["<tool>", ...]}

  statistical:    {"metric": "latency_ms", "tool": "<tool>", "threshold": <ms>}
                  | {"metric": "success_rate", "tool": "<tool>", "rate": <0..1>}
                  | {"metric": "feature_envelope", "feature": "<name>",
                     "p1": <number>, "p99": <number>}

  state:          {"terminator_tool": "<tool>", "forbidden_after": ["<tool>", ...]}

  composition:    {"upstream_tool": "<tool>", "downstream_tool": "<tool>"}

  tool_selection: {"segment": "<segment id>", "expected_tool": "<tool>",
                   "forbidden_tool": "<tool or null>"}

  output_format:  {"rubric": "<rubric for LLM-as-judge>"}

Rules for high-quality output:
1. Only emit invariants that hold across MULTIPLE traces in the batch.
2. Prefer SURPRISING rules; skip the obvious.
3. Tool names in `params` must match the trace `tool` field exactly.
4. Cite at least 3 trace_ids per invariant in `evidence_trace_ids`.
5. Use plain English in `description` — no temporal-logic operators.
6. Cover diverse types — don't return only orderings.
7. If you cannot fit a behavior into a `params` shape, skip it.
"""


def _summarize_step(step: Any) -> dict[str, Any]:
    """Compact a step for prompting; keep the discriminating fields only."""
    s = step.model_dump() if hasattr(step, "model_dump") else dict(step)
    out: dict[str, Any] = {"type": s.get("type")}
    for key in ("content", "tool", "args", "latency_ms", "result_status"):
        if key in s and s[key] is not None:
            value = s[key]
            if isinstance(value, str) and len(value) > 240:
                value = value[:240] + "…"
            out[key] = value
    return out


def build_user_message(traces: list[NormalizedTrace]) -> str:
    """Render a trace batch into a compact, model-friendly JSON payload."""
    summary = [
        {
            "trace_id": t.trace_id,
            "metadata": t.metadata.model_dump(exclude_none=True),
            "steps": [_summarize_step(step) for step in t.steps],
        }
        for t in traces
    ]
    payload = {
        "task": (
            "Identify behavioral invariants. What rules does this agent "
            "consistently follow? What does it never do?"
        ),
        "n_traces": len(traces),
        "traces": summary,
    }
    return json.dumps(payload, ensure_ascii=False)
