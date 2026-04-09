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
