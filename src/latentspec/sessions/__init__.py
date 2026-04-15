"""Session-level invariants: rules across multi-turn agent sessions.

A `Session` groups N turns (each turn is a `NormalizedTrace`) keyed by
`session_id`. Some behavioral rules only manifest at this scale:

  - "After a `refund` turn, the next turn always involves
    `customer_followup`."
  - "A session never contains more than one `payments_v2` call."
  - "Every session ends with a `session_close` turn."

This is a separate orchestrator that runs alongside per-turn mining.
Turn-level rules constrain individual traces; session-level rules
constrain the *transitions and aggregates* between traces.
"""

from latentspec.sessions.orchestrator import (
    SessionMiningResult,
    mine_session_invariants,
)
from latentspec.sessions.schema import Session, SessionTransition

__all__ = [
    "Session",
    "SessionMiningResult",
    "SessionTransition",
    "mine_session_invariants",
]
