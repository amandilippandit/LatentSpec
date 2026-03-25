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
                description=item["description"],
                formal_rule=item.get("formal_rule") or "",
                severity=Severity(item.get("severity", "medium")),
                params=item.get("params") or {},
            )
        )
    return out


@cli.command()
@click.option(
    "--invariants",
    "invariants_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSON list of invariants (output of `latentspec mine --out …`).",
)
@click.option(
    "--baseline",
    "baseline_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSON list of baseline traces (the known-good population).",
)
@click.option(
    "--candidate",
    "candidate_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSON list of candidate traces (the new population to check).",
)
@click.option("--agent-name", default="agent", help="Used in the report header.")
@click.option(
    "--fail-on",
    type=click.Choice(["never", "warn", "any", "high", "critical"]),
    default="critical",
    help="Severity gate that controls the exit code.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["pr", "terminal", "json"]),
    default="terminal",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the rendered report to this file (in addition to stdout).",
)
def check(
    invariants_path: Path,
    baseline_path: Path,
    candidate_path: Path,
    agent_name: str,
    fail_on: str,
    output_format: str,
    output_path: Path | None,
) -> None:
    """Run a behavioral regression check (§4.1 batch + §4.2 PR comment).

    Returns exit 1 when the configured severity gate is breached, exit 0
    otherwise — making it directly usable as a CI step.
    """
    invariants = _load_invariants(invariants_path)
    baseline = _load_traces(baseline_path)
    candidate = _load_traces(candidate_path)

    console.print(
        f"Checking [bold]{len(invariants)}[/bold] invariants "
        f"against [bold]{len(baseline)}[/bold] baseline + "
        f"[bold]{len(candidate)}[/bold] candidate traces…"
    )
    report = compare_trace_sets(invariants, baseline, candidate)

    if output_format == "json":
        payload = {
            "invariants_checked": report.invariants_checked,
            "passes": report.passes,
            "warnings": [_summary_dict(s) for s in report.warnings],
            "failures": [_summary_dict(s) for s in report.failures],
            "counts": report.counts,
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    elif output_format == "pr":
        text = format_pr_comment(report, agent_name=agent_name)
    else:
        text = format_terminal(report, agent_name=agent_name)

    click.echo(text)
    if output_path is not None:
        output_path.write_text(text)
        console.print(f"[dim]Wrote report to {output_path}[/dim]")

    from latentspec.regression.batch import _exit_code_for

    sys.exit(_exit_code_for(report, fail_on))


# ----- export-promptfoo (week 3) -----------------------------------------


@cli.command(name="export-promptfoo")
@click.option(
    "--invariants",
    "invariants_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSON list of invariants (output of `latentspec mine --out …`).",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to write the generated promptfoo.yaml.",
)
@click.option(
    "--include",
    type=click.Choice(["active", "all"]),
    default="active",
    help="Which invariants to include: only active ones, or every status.",
)
def export_promptfoo_cmd(
    invariants_path: Path, out_path: Path | None, include: str
) -> None:
    """Generate a promptfoo.yaml from a discovered invariant set (§2.2)."""
    raw = json.loads(invariants_path.read_text())
    if not isinstance(raw, list):
        raise click.ClickException(f"{invariants_path} must contain a JSON list")

    invariants: list[MinedInvariant] = [MinedInvariant.model_validate(item) for item in raw]
    if include == "active":
        invariants = [
            inv for inv in invariants if inv.status == InvariantStatus.ACTIVE
        ]

    yaml_text = export_promptfoo(invariants)
    if out_path is not None:
        out_path.write_text(yaml_text)
        console.print(
            f"[green]Exported {len(invariants)} invariants to {out_path}[/green]"
        )
    else:
        click.echo(yaml_text)


def _summary_dict(s) -> dict:
    return {
        "invariant_id": s.invariant_id,
        "type": s.type.value,
        "description": s.description,
        "severity": s.severity.value,
        "applicable": s.applicable,
        "passed": s.passed,
        "failed": s.failed,
        "warned": s.warned,
        "pass_rate": s.pass_rate,
        "fail_rate": s.fail_rate,
        "warn_rate": s.warn_rate,
        "sample_failure_traces": s.sample_failure_traces,
    }


if __name__ == "__main__":
    cli()
