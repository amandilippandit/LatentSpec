"""Per-agent version tracking + tool-repertoire diffing.

Called from the trace ingest path: when a `version_tag` shows up that we
haven't seen for this agent, we create an `AgentVersion` row and seed it
with the trace's tool set. Subsequent traces with the same tag update
`last_seen_at` and union new tools into the recorded repertoire.

`diff_versions(old, new)` returns a `VersionDelta` describing added /
removed / renamed tools — useful for the dashboard, for migration prompts
("you renamed `payments_v1` to `payments_v2` — should we migrate the rule
set?"), and for the canonicaliser which can use it as a strong signal.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from latentspec.models import AgentVersion
from latentspec.schemas.trace import NormalizedTrace, ToolCallStep

log = logging.getLogger(__name__)


@dataclass
class VersionDelta:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    common: list[str] = field(default_factory=list)
    likely_renames: list[tuple[str, str]] = field(default_factory=list)

    @property
    def is_breaking(self) -> bool:
        """Heuristic — > 30% repertoire churn flagged as breaking."""
        old_size = len(self.removed) + len(self.common)
        if old_size == 0:
            return False
        churn = (len(self.added) + len(self.removed)) / max(1, old_size)
        return churn > 0.3


def _trace_repertoire(trace: NormalizedTrace) -> set[str]:
    return {s.tool for s in trace.steps if isinstance(s, ToolCallStep)}


async def register_or_update_version(
    db: AsyncSession,
    *,
    agent_id: uuid.UUID,
    version_tag: str,
    trace: NormalizedTrace,
    parent_version_tag: str | None = None,
) -> AgentVersion:
    """Idempotent — first call inserts, subsequent calls union tools + bump last_seen."""
    row = (
        await db.execute(
            select(AgentVersion)
            .where(AgentVersion.agent_id == agent_id)
            .where(AgentVersion.version_tag == version_tag)
        )
    ).scalar_one_or_none()

    new_tools = _trace_repertoire(trace)

    if row is None:
        row = AgentVersion(
            agent_id=agent_id,
            version_tag=version_tag,
            tool_repertoire=sorted(new_tools),
            parent_version_tag=parent_version_tag,
        )
        db.add(row)
        await db.flush()
        return row

    union = sorted(set(row.tool_repertoire or []) | new_tools)
    if union != list(row.tool_repertoire or []):
        row.tool_repertoire = union
    row.last_seen_at = datetime.now(UTC)
    return row


async def resolve_active_version(
    db: AsyncSession, *, agent_id: uuid.UUID
) -> AgentVersion | None:
    """Return the agent's most-recently-seen version row, or None."""
    return (
        await db.execute(
            select(AgentVersion)
            .where(AgentVersion.agent_id == agent_id)
            .order_by(AgentVersion.last_seen_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


def diff_versions(old: Iterable[str], new: Iterable[str]) -> VersionDelta:
    """Tool-set diff with edit-distance-based rename detection."""
    from latentspec.canonicalization.canonicalizer import _levenshtein, canonical_form

    old_set = set(old)
    new_set = set(new)
    added = sorted(new_set - old_set)
    removed = sorted(old_set - new_set)
    common = sorted(old_set & new_set)

    likely_renames: list[tuple[str, str]] = []
    used_added: set[str] = set()
    for old_name in removed:
        best = None
        best_dist = 999
        old_canon = canonical_form(old_name)
        for cand in added:
            if cand in used_added:
                continue
            cand_canon = canonical_form(cand)
            min_len = min(len(old_canon), len(cand_canon))
            max_len = max(len(old_canon), len(cand_canon))
            # Renames only meaningful for non-trivial names
            if min_len < 4 or max_len < 4:
                continue
            d = _levenshtein(old_canon, cand_canon, max_distance=3)
            # Relative threshold: edit distance ≤ 25% of the longer name AND
            # an absolute floor of 2 so we don't flag random 1-char swaps.
            if d <= 2 and d <= max(1, max_len // 4) and d < best_dist:
                best_dist = d
                best = cand
        if best is not None:
            likely_renames.append((old_name, best))
            used_added.add(best)

    return VersionDelta(
        added=[a for a in added if a not in used_added],
        removed=[r for r in removed if not any(r == old for old, _ in likely_renames)],
        common=common,
        likely_renames=likely_renames,
    )
