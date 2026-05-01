"""Session and SessionTransition Pydantic schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from latentspec.schemas.trace import NormalizedTrace


class SessionTransition(BaseModel):
    """One turn-to-turn transition derived from `Session.turns`."""

    model_config = ConfigDict(extra="allow")

    from_fingerprint: str
    to_fingerprint: str
    from_trace_id: str
    to_trace_id: str
    elapsed_seconds: float | None = None


class Session(BaseModel):
    """N ordered turns sharing a session id."""

    model_config = ConfigDict(extra="allow")

    session_id: str
    agent_id: str
    user_id: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    turns: list[NormalizedTrace]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    def transitions(self) -> list[SessionTransition]:
        from latentspec.mining.fingerprint import fingerprint

        out: list[SessionTransition] = []
        for prev, nxt in zip(self.turns, self.turns[1:], strict=False):
            elapsed: float | None = None
            if prev.ended_at and nxt.timestamp:
                elapsed = max(0.0, (nxt.timestamp - prev.ended_at).total_seconds())
            out.append(
                SessionTransition(
                    from_fingerprint=fingerprint(prev),
                    to_fingerprint=fingerprint(nxt),
                    from_trace_id=prev.trace_id,
                    to_trace_id=nxt.trace_id,
                    elapsed_seconds=elapsed,
                )
            )
        return out
