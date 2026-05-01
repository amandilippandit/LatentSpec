"""HITL review queue for synthetic traces.

The queue is in-memory + JSON-serializable so it can be persisted, sent
to a UI, or shipped to a human-review service. Each `SyntheticTrace`
carries provenance (which `AgentSpec` produced it, which prompt) so the
audit trail is complete: a regulator can ask "where did this rule come
from?" and we can trace it back to the seed spec.
"""

from __future__ import annotations

import enum
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from latentspec.schemas.trace import NormalizedTrace


class ReviewDecision(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"


@dataclass
class SyntheticTrace:
    """A synthetic trace plus provenance and review state."""

    id: str
    trace: NormalizedTrace
    spec_name: str
    decision: ReviewDecision = ReviewDecision.PENDING
    decided_at: datetime | None = None
    decided_by: str | None = None
    edit_notes: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "spec_name": self.spec_name,
            "decision": self.decision.value,
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "decided_by": self.decided_by,
            "edit_notes": self.edit_notes,
            "created_at": self.created_at.isoformat(),
            "trace": self.trace.model_dump(mode="json"),
        }


@dataclass
class ReviewQueue:
    """Process-wide queue of pending synthetic traces."""

    items: dict[str, SyntheticTrace] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def submit(
        self, trace: NormalizedTrace, *, spec_name: str
    ) -> SyntheticTrace:
        item = SyntheticTrace(
            id=f"synth-{uuid.uuid4().hex[:12]}",
            trace=trace,
            spec_name=spec_name,
        )
        with self._lock:
            self.items[item.id] = item
        return item

    def submit_many(
        self, traces: list[NormalizedTrace], *, spec_name: str
    ) -> list[SyntheticTrace]:
        return [self.submit(t, spec_name=spec_name) for t in traces]

    def pending(self) -> list[SyntheticTrace]:
        with self._lock:
            return [
                i for i in self.items.values() if i.decision == ReviewDecision.PENDING
            ]

    def approved_traces(self) -> list[NormalizedTrace]:
        with self._lock:
            return [
                i.trace
                for i in self.items.values()
                if i.decision in (ReviewDecision.APPROVED, ReviewDecision.EDITED)
            ]

    def decide(
        self,
        item_id: str,
        *,
        decision: ReviewDecision,
        decided_by: str | None = None,
        edit_notes: str | None = None,
        replacement_trace: NormalizedTrace | None = None,
    ) -> SyntheticTrace:
        with self._lock:
            item = self.items.get(item_id)
            if item is None:
                raise KeyError(f"unknown synthetic trace: {item_id}")
            item.decision = decision
            item.decided_at = datetime.now(UTC)
            item.decided_by = decided_by
            item.edit_notes = edit_notes
            if replacement_trace is not None and decision == ReviewDecision.EDITED:
                item.trace = replacement_trace
            return item

    def stats(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {d.value: 0 for d in ReviewDecision}
            for i in self.items.values():
                counts[i.decision.value] += 1
            counts["total"] = len(self.items)
            return counts
