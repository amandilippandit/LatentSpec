"""LatentSpec command-line interface.

Commands:
  latentspec mine    --traces FILE         run mining locally
  latentspec demo                          synthetic week-1 milestone demo
  latentspec check   --baseline B --candidate C --invariants I  (week-2)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from latentspec.checking.base import InvariantSpec
from latentspec.exporters import export_promptfoo
from latentspec.mining.orchestrator import mine_invariants
from latentspec.models.invariant import InvariantStatus, InvariantType, Severity
from latentspec.regression.batch import compare_trace_sets
from latentspec.regression.report import format_pr_comment, format_terminal
from latentspec.schemas.invariant import MinedInvariant
from latentspec.schemas.trace import NormalizedTrace

console = Console()


@click.group()
@click.version_option()
def cli() -> None:
    """LatentSpec — discover behavioral invariants from AI agent traces."""


# ----- mine ---------------------------------------------------------------


@cli.command()
@click.option(
    "--traces",
    "traces_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSON file containing a list of §3.2-shaped traces.",
)
@click.option("--no-llm", is_flag=True, help="Skip Track B (LLM mining).")
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to save the discovered invariant set as JSON "
    "(suitable for `latentspec check --invariants`).",
)
def mine(traces_path: Path, no_llm: bool, out_path: Path | None) -> None:
    """Run the §3.2 mining pipeline against a local trace file."""
    if no_llm:
        os.environ["ANTHROPIC_API_KEY"] = ""

    raw = json.loads(traces_path.read_text())
    if not isinstance(raw, list):
        click.echo("traces file must contain a JSON list of traces", err=True)
        sys.exit(2)

    traces = [NormalizedTrace.model_validate(item) for item in raw]
    console.print(f"Loaded [bold]{len(traces)}[/bold] traces from {traces_path}")

    result = asyncio.run(
        mine_invariants(
            agent_id=uuid.uuid4(),
            traces=traces,
            session=None,
            persist=False,
        )
    )

    table = Table(title=f"Discovered {len(result.invariants)} invariants")
    table.add_column("Type", style="cyan")
    table.add_column("Sev", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Conf", justify="right")
    table.add_column("Source")
    table.add_column("Description")

    for inv in result.invariants:
        table.add_row(
            inv.type.value,
            inv.severity.value,
            inv.status.value,
            f"{inv.confidence:.2f}",
            inv.discovered_by,
            inv.description[:80] + ("…" if len(inv.description) > 80 else ""),
        )
    console.print(table)
    console.print(
        f"Statistical: {result.candidates_statistical}  "
        f"LLM: {result.candidates_llm}  "
        f"Unique merged: {result.candidates_total_unique}  "
        f"Duration: {result.duration_seconds:.2f}s"
    )

    if out_path is not None:
        payload = [inv.model_dump(mode="json") for inv in result.invariants]
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        console.print(f"[green]Saved {len(payload)} invariants to {out_path}[/green]")


# ----- demo ---------------------------------------------------------------


@cli.command()
@click.option("--n-traces", type=int, default=240)
@click.option("--no-llm", is_flag=True)
def demo(n_traces: int, no_llm: bool) -> None:
    """Run the §9 week-1 milestone demo (synthetic booking-agent)."""
    from latentspec.demo import generate_traces

    if no_llm:
        os.environ["ANTHROPIC_API_KEY"] = ""

    traces = generate_traces(n_traces)
    result = asyncio.run(
        mine_invariants(
            agent_id=uuid.uuid4(), traces=traces, session=None, persist=False
        )
    )

    console.print(
        f"\nDemo: [bold]{len(traces)}[/bold] traces -> "
        f"[bold green]{len(result.invariants)}[/bold green] invariants "
        f"across {len(result.by_type)} types"
    )
    for inv in result.invariants:
        console.print(
            f"  [dim]{inv.severity.value:>8}[/dim] "
            f"[cyan]{inv.type.value:>14}[/cyan] "
            f"conf=[bold]{inv.confidence:.2f}[/bold] "
            f"{inv.description}"
        )


# ----- check (week 2) -----------------------------------------------------


def _load_traces(path: Path) -> list[NormalizedTrace]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise click.ClickException(f"{path} must contain a JSON list of traces")
    return [NormalizedTrace.model_validate(item) for item in raw]


def _load_invariants(path: Path) -> list[InvariantSpec]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise click.ClickException(f"{path} must contain a JSON list of invariants")
    out: list[InvariantSpec] = []
    for item in raw:
        out.append(
            InvariantSpec(
                id=str(item.get("invariant_id") or item.get("id") or uuid.uuid4().hex),
                type=InvariantType(item["type"]),
