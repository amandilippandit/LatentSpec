"""Run the LatentSpec pipeline against three real-shape corpora.

For each corpus:
  1. Generate N traces.
  2. Run the calibrator → get per-corpus thresholds.
  3. Run the miner with those thresholds.
  4. Print: tool repertoire, n distinct fingerprints, n invariants by
     type, top 5 highest-confidence rules.

This is the script that surfaces real failures. Anything that breaks
here is a real gap in the pipeline that the synthetic booking-agent
demo + my own test fixtures hid.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from collections import Counter
from pathlib import Path

# Allow direct invocation
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "")  # disable Track B for the harness

from latentspec.calibration.calibrator import calibrate_agent
from latentspec.canonicalization.canonicalizer import (
    ToolCanonicalizer,
    collect_tool_names,
)
from latentspec.canonicalization.applier import apply_canonicalisation
from latentspec.mining.fingerprint import fingerprint
from latentspec.mining.orchestrator import mine_invariants
from latentspec.schemas.trace import ToolCallStep
from scripts.realcorpus import autogpt_corpus, babyagi_corpus, opendevin_corpus


CORPORA = [
    ("BabyAGI", babyagi_corpus, 200),
    ("OpenDevin", opendevin_corpus, 200),
    ("AutoGPT", autogpt_corpus, 200),
]


async def run_one(name: str, module, n_traces: int) -> None:
    print()
    print("=" * 64)
    print(f"  Corpus: {name}  (n_traces={n_traces})")
    print("=" * 64)

    traces = module.generate(n_traces)

    # Tool inventory + canonicalisation
    raw_tools = collect_tool_names(traces)
    cano = ToolCanonicalizer().fit(raw_tools)
    canonical_traces = [apply_canonicalisation(t, cano) for t in traces]
    canonical_tool_count = len(cano.clusters)

    fp_counts: Counter[str] = Counter()
    step_counts = []
    for t in canonical_traces:
        fp_counts[fingerprint(t)] += 1
        step_counts.append(len(t.steps))

    print(
        f"  raw tool names              : {len(raw_tools)}\n"
        f"  canonical tool clusters     : {canonical_tool_count}\n"
        f"  distinct trace fingerprints : {len(fp_counts)}\n"
        f"  median trace length         : {sorted(step_counts)[len(step_counts)//2]}\n"
        f"  p99 trace length            : {sorted(step_counts)[int(0.99*len(step_counts))]}\n"
        f"  top fingerprint share       : {max(fp_counts.values()) / n_traces:.1%}"
    )

    # Calibrate thresholds
    th = calibrate_agent(canonical_traces)
    print(
        f"\n  calibrated min_support      : {th.mining_min_support:.3f}\n"
        f"  calibrated max_path_length  : {th.mining_max_path_length}\n"
        f"  calibrated review_threshold : {th.confidence_review_threshold:.3f}\n"
        f"  calibrated chi-square thr   : {th.fingerprint_chi_square_threshold:.2f}"
    )

    # Mine
    result = await mine_invariants(
        agent_id=uuid.uuid4(), traces=canonical_traces, session=None, persist=False
    )
    print(f"\n  mining produced {len(result.invariants)} invariants:")
    for type_name, count in sorted(result.by_type.items()):
        print(f"    {type_name:>16}: {count}")

    # Show top 5 by confidence
    if result.invariants:
        top = sorted(result.invariants, key=lambda i: -i.confidence)[:5]
        print("\n  top-5 by confidence:")
        for inv in top:
            print(
                f"    [{inv.severity.value:>8}] conf={inv.confidence:.2f} "
                f"({inv.type.value:>12}) {inv.description[:90]}"
            )


async def main() -> None:
    for name, module, n in CORPORA:
        try:
            await run_one(name, module, n)
        except Exception as e:
            print(f"\n!! {name} pipeline RAISED: {e!r}")


if __name__ == "__main__":
    asyncio.run(main())
