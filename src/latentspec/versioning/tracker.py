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
