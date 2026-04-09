"""Vertical pack loader + auto-fit scoring.

A pack is a JSON file shipping a list of invariants in the same Pydantic
shape mining produces (validated against `validate_params(type, params)`).
`install_pack(agent_id, pack_id)` reads the file, validates each entry,
and emits a list of `MinedInvariant` objects ready to be persisted.

`auto_fit_score(invariant, traces)` runs the rule against a sample of
the agent's actual traces and returns a fit score in `[0, 1]` based on:

  - **applicability** — fraction of traces where the rule's preconditions
    apply
  - **pass rate** — fraction of applicable traces where the rule holds
  - **severity-weighted blend** — critical rules stay even at marginal
    fit, low-severity rules drop fast

The score lets us auto-promote / auto-demote pack rules without human
review.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import resources
from typing import Any

from latentspec.checking.base import CheckOutcome, InvariantSpec
from latentspec.checking.dispatch import dispatch
from latentspec.models.invariant import (
    InvariantStatus,
    InvariantType,
    Severity,
)
from latentspec.schemas.invariant import MinedInvariant
from latentspec.schemas.params import ParamsValidationError, validate_params
from latentspec.schemas.trace import NormalizedTrace

log = logging.getLogger(__name__)


@dataclass
class PackInvariant:
    """Pre-built invariant from a vertical pack."""

    type: InvariantType
    description: str
    formal_rule: str
    severity: Severity
    params: dict[str, Any]
    rationale: str = ""
    regulatory_refs: list[str] = field(default_factory=list)


@dataclass
class VerticalPack:
    """Collection of `PackInvariant` for a specific vertical."""

    pack_id: str
    title: str
    description: str
    version: str
    invariants: list[PackInvariant] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    regulatory_frameworks: list[str] = field(default_factory=list)


# ---- pack registry -------------------------------------------------------


_PACK_PACKAGE = "latentspec.packs.data"


def _load_pack_file(pack_id: str) -> dict[str, Any] | None:
    try:
        files = resources.files(_PACK_PACKAGE)
        path = files.joinpath(f"{pack_id}.json")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ModuleNotFoundError, AttributeError):
        return None


def _parse_pack(raw: dict[str, Any]) -> VerticalPack:
    invariants: list[PackInvariant] = []
    for inv in raw.get("invariants", []):
        try:
            inv_type = InvariantType(inv["type"])
            params = validate_params(inv_type, dict(inv.get("params") or {}))
        except (KeyError, ValueError, ParamsValidationError) as e:
            log.warning(
                "skipping pack invariant in %s due to %s", raw.get("pack_id"), e
            )
            continue
        invariants.append(
            PackInvariant(
                type=inv_type,
                description=str(inv["description"]),
                formal_rule=str(inv.get("formal_rule") or ""),
                severity=Severity(inv.get("severity", "medium")),
                params=params,
                rationale=str(inv.get("rationale", "")),
                regulatory_refs=list(inv.get("regulatory_refs", [])),
            )
        )
    return VerticalPack(
        pack_id=raw["pack_id"],
        title=raw.get("title", raw["pack_id"].title()),
        description=raw.get("description", ""),
        version=raw.get("version", "0.1.0"),
        invariants=invariants,
        targets=list(raw.get("targets", [])),
        regulatory_frameworks=list(raw.get("regulatory_frameworks", [])),
    )


def get_pack(pack_id: str) -> VerticalPack | None:
    raw = _load_pack_file(pack_id)
    if raw is None:
        return None
    return _parse_pack(raw)


def list_packs() -> list[str]:
    try:
        files = resources.files(_PACK_PACKAGE)
        return sorted(
            p.name.removesuffix(".json")
            for p in files.iterdir()
            if p.name.endswith(".json")
        )
    except (ModuleNotFoundError, AttributeError):
        return []


# ---- install + score -----------------------------------------------------


def install_pack(
    *, agent_id: uuid.UUID, pack_id: str
) -> list[MinedInvariant]:
    """Materialize a pack into a list of `MinedInvariant` rows.

    Each materialized invariant carries `discovered_by="pack"`, a stable
    `cluster_id`-style `pack_id` in params, and starts in `status=pending`
    until `auto_fit_score` validates it against real traffic.
    """
    pack = get_pack(pack_id)
    if pack is None:
        raise ValueError(f"unknown pack: {pack_id}")

    materialized: list[MinedInvariant] = []
    now = datetime.now(UTC)
    for entry in pack.invariants:
        try:
            # Validate FIRST against the strict per-type schema so we can't
            # ship a pack rule the runtime checker can't read. Provenance
            # (pack_id, pack_version) is added afterwards.
            params = validate_params(entry.type, dict(entry.params))
        except ParamsValidationError:
            continue
        params = {
            **params,
            "pack_id": pack.pack_id,
            "pack_version": pack.version,
        }
        materialized.append(
            MinedInvariant(
                invariant_id=f"pack-{uuid.uuid4().hex[:8]}",
                type=entry.type,
                description=entry.description,
                formal_rule=entry.formal_rule
                or f"pack({pack.pack_id}).{entry.type.value}",
                confidence=0.65,  # mid-band — pending until auto-fit promotes
                support_score=0.7,
                consistency_score=0.7,
                cross_val_bonus=0.0,
                clarity_score=0.85,
                evidence_count=0,
                violation_count=0,
                discovered_at=now,
                status=InvariantStatus.PENDING,
                severity=entry.severity,
                discovered_by="pack",  # type: ignore[arg-type]
                evidence_trace_ids=[],
                params=params,
            )
        )
    return materialized


@dataclass
class AutoFitScore:
    fit: float  # weighted final score in [0, 1]
    applicability: float  # fraction of traces where the rule applied
    pass_rate: float  # fraction of applicable traces where it held
    sample_size: int


_SEVERITY_WEIGHT = {
    Severity.CRITICAL: 0.4,  # critical rules stay even at marginal fit
    Severity.HIGH: 0.5,
    Severity.MEDIUM: 0.6,
    Severity.LOW: 0.7,
}


def auto_fit_score(
    *,
    invariant: MinedInvariant,
    traces: list[NormalizedTrace],
) -> AutoFitScore:
    """Score how well a pack invariant fits the agent's real traces.

    Applicability + pass rate combine via a severity-weighted threshold:
    higher severity makes the rule sticker (we'd rather keep a critical
    rule pending review than auto-reject it).
    """
    spec = InvariantSpec(
        id=invariant.invariant_id,
        type=invariant.type,
        description=invariant.description,
        formal_rule=invariant.formal_rule,
        severity=invariant.severity,
        params={k: v for k, v in invariant.params.items() if k not in {"pack_id", "pack_version"}},
    )

    n_applicable = 0
    n_passing = 0
    for trace in traces:
        outcome = dispatch(spec, trace).outcome
        if outcome == CheckOutcome.NOT_APPLICABLE:
            continue
        n_applicable += 1
        if outcome == CheckOutcome.PASS:
            n_passing += 1

    if not traces:
        return AutoFitScore(fit=0.0, applicability=0.0, pass_rate=0.0, sample_size=0)

    applicability = n_applicable / len(traces)
    pass_rate = n_passing / max(1, n_applicable)
    weight = _SEVERITY_WEIGHT[invariant.severity]
    fit = weight * applicability + (1 - weight) * pass_rate

    return AutoFitScore(
        fit=round(fit, 4),
        applicability=round(applicability, 4),
        pass_rate=round(pass_rate, 4),
        sample_size=len(traces),
    )
