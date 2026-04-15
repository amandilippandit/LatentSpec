"""Session-level mining: discover cross-turn invariants.

Three families of session-level rules we mine:

  1. **Transition rules** — Markov-chain style: P(next_fingerprint |
     current_fingerprint). When P(B|A) >= threshold and the support is
     high enough, we emit "after a turn of shape A, the next turn has
     shape B" as a composition invariant.

  2. **Aggregate rules** — bounded counts of specific tools across all
     turns of a session. Used for rules like "≤ 1 `payments_v2` call per
     session" or "≥ 1 `auth_user` call per session".

  3. **Termination rules** — fingerprint distribution of last turns.
     Catches "every session ends with a `farewell` shape" and similar.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from latentspec.mining.fingerprint import fingerprint
from latentspec.models.invariant import InvariantType, Severity
from latentspec.schemas.invariant import InvariantCandidate
from latentspec.schemas.trace import ToolCallStep
from latentspec.sessions.schema import Session

log = logging.getLogger(__name__)


@dataclass
class SessionMiningResult:
    n_sessions: int
    transitions: list[InvariantCandidate] = field(default_factory=list)
    aggregates: list[InvariantCandidate] = field(default_factory=list)
    terminations: list[InvariantCandidate] = field(default_factory=list)

    def all(self) -> list[InvariantCandidate]:
        return [*self.transitions, *self.aggregates, *self.terminations]


def _mine_transitions(
    sessions: Sequence[Session],
    *,
    min_support: float = 0.4,
    min_conditional_prob: float = 0.85,
) -> list[InvariantCandidate]:
    """For each (from_fp, to_fp) pair, count co-occurrence across sessions.

    Emit composition invariants when:
      - P(to_fp | from_fp) >= `min_conditional_prob`
      - support across sessions >= `min_support`
    """
    pair_count: Counter[tuple[str, str]] = Counter()
    from_count: Counter[str] = Counter()
    pair_session_evidence: dict[tuple[str, str], set[str]] = defaultdict(set)
    n = max(1, len(sessions))

    for session in sessions:
        for tr in session.transitions():
            pair = (tr.from_fingerprint, tr.to_fingerprint)
            pair_count[pair] += 1
            from_count[tr.from_fingerprint] += 1
            pair_session_evidence[pair].add(session.session_id)

    out: list[InvariantCandidate] = []
    for (from_fp, to_fp), c in pair_count.items():
        denom = from_count[from_fp]
        if denom == 0:
            continue
        cond_p = c / denom
        support = len(pair_session_evidence[(from_fp, to_fp)]) / n
        if cond_p < min_conditional_prob or support < min_support:
            continue
        out.append(
            InvariantCandidate(
                type=InvariantType.COMPOSITION,
                description=(
                    f"After a turn with shape `{from_fp[:8]}`, the next turn "
                    f"has shape `{to_fp[:8]}` ({cond_p:.0%} of the time)"
                ),
                formal_rule=(
                    f"forall session, t in session.turns "
                    f"where fp(t) == '{from_fp}': "
                    f"P(fp(next(t)) == '{to_fp}') >= {min_conditional_prob}"
                ),
                evidence_trace_ids=[],
                support=round(support, 4),
                consistency=round(cond_p, 4),
                severity=Severity.MEDIUM,
                discovered_by="statistical",
                extra={
                    "upstream_tool": f"fingerprint:{from_fp[:8]}",
                    "downstream_tool": f"fingerprint:{to_fp[:8]}",
                    "from_fingerprint": from_fp,
                    "to_fingerprint": to_fp,
                    "session_level": True,
                    "transition_probability": round(cond_p, 4),
                },
            )
        )
    return out


def _mine_aggregates(
    sessions: Sequence[Session],
    *,
    upper_bound_quantile: float = 0.99,
    min_consistency: float = 0.95,
) -> list[InvariantCandidate]:
    """Per-tool aggregate counts across each session.

    Emits a STATISTICAL invariant (`metric=feature_envelope`) for every
    tool whose per-session count distribution is tight, e.g. "≤ 1
