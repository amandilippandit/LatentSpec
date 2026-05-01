"""Vertical invariant packs — bootstrap a new agent with industry-standard rules.

Cold-start problem: an agent with no production history has no patterns
to mine, so the rule set starts empty. Vertical packs solve this by
shipping curated invariant bundles for specific verticals (e-commerce,
banking, healthcare) that encode regulatory + best-practice expectations.

Workflow:

  1. Customer registers an agent.
  2. They install one or more packs (`POST /agents/{id}/packs/install`).
  3. Pack invariants enter the agent's invariant set with `status=pending`.
  4. As real traces arrive, `auto_fit_score` measures how well each pack
     rule matches the agent's actual behavior. High-fit rules auto-promote
     to `active`; low-fit rules auto-demote to `rejected` (with audit).
  5. Mining continues to discover agent-specific rules on top.

The packs themselves are JSON files in `latentspec/packs/data/`. Each
file contains a list of invariants in the same shape mining produces.
"""

from latentspec.packs.library import (
    PackInvariant,
    VerticalPack,
    auto_fit_score,
    get_pack,
    install_pack,
    list_packs,
)

__all__ = [
    "PackInvariant",
    "VerticalPack",
    "auto_fit_score",
    "get_pack",
    "install_pack",
    "list_packs",
]
