"""§9 week-1 milestone runner.

Generates 200+ synthetic booking-agent traces, runs the §3.2 mining pipeline,
and reports whether the milestone gate (≥10 invariants across ≥3 types) passes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

# Allow direct invocation without installing the package
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latentspec.demo import generate_traces  # noqa: E402
from latentspec.mining.orchestrator import mine_invariants  # noqa: E402

log = logging.getLogger("latentspec.demo")


def _format_invariants(invariants: list, *, type_breakdown: dict[str, int]) -> str:
    lines: list[str] = []
    lines.append("\n=== LatentSpec Mining Result (§9 week-1 milestone) ===")
    lines.append(
        f"Discovered {len(invariants)} invariants across {len(type_breakdown)} types\n"
    )
    for inv in invariants:
        lines.append(
            f"  [{inv.severity.value:>8}] "
            f"{inv.type.value:>14} | conf={inv.confidence:.2f} "
            f"({inv.discovered_by:>10}) status={inv.status.value}"
        )
        lines.append(f"             {inv.description}")
    lines.append("\nBreakdown by type:")
    for t, c in sorted(type_breakdown.items()):
        lines.append(f"  {t:>14}: {c}")
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description="LatentSpec week-1 milestone demo")
    parser.add_argument("--n-traces", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="skip Track B (LLM) — useful when no API key is set",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional path to dump generated traces as JSON",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    )

    log.info("Generating %d synthetic booking-agent traces…", args.n_traces)
    traces = generate_traces(args.n_traces, seed=args.seed)

    if args.out:
        args.out.write_text(
            json.dumps(
                [t.model_dump(mode="json") for t in traces],
                ensure_ascii=False,
                indent=2,
            )
        )
        log.info("Wrote %d traces to %s", len(traces), args.out)

    if args.no_llm:
        import os

        os.environ["ANTHROPIC_API_KEY"] = ""

    agent_id = uuid.uuid4()
    log.info("Running mining pipeline on agent %s…", agent_id)
    result = await mine_invariants(
        agent_id=agent_id,
        traces=traces,
        session=None,
        persist=False,
    )

    print(_format_invariants(result.invariants, type_breakdown=result.by_type))
    print(
        f"\nStatistical candidates: {result.candidates_statistical} | "
        f"LLM candidates: {result.candidates_llm} | "
        f"Unique merged: {result.candidates_total_unique}"
    )
    print(f"Duration: {result.duration_seconds:.2f}s")
    print(f"Status breakdown: {result.by_status}")

    n = len(result.invariants)
    n_types = len(result.by_type)
    if n >= 10 and n_types >= 3:
        print(
            f"\n[PASS] Milestone — {n} invariants across {n_types} types "
            "(target: >=10 invariants, >=3 types)"
        )
        return 0
    print(
        f"\n[FAIL] Milestone — {n} invariants across {n_types} types "
        "(target: >=10 invariants, >=3 types)"
    )
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
