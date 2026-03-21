"""Tool-name canonicalisation.

Real agents emit the same logical tool under multiple names:
  payments_v1, payments_v2, payments-v2, Payments_v2, payments.v2.execute

Without canonicalisation:
  - Mining produces N rules instead of 1.
  - Closed-world repertoire fires false positives whenever an agent
    renames a tool.
  - Cluster routing breaks because vectoriser-time vs runtime tool sets
    diverge.

The canonicaliser learns alias clusters per agent in three passes,
strongest signal first:

  1. **Normalised exact match** — strip punctuation, lowercase, drop
     versioning suffixes. Catches casing / punctuation / version drift.
  2. **Token-Jaccard + edit distance** — `book_flight` vs `flight.book`
     share the {book, flight} token set; `delete_user` vs `delete-users`
     have edit distance 2. Catches morphology + ordering.
  3. **Character n-gram cosine** — fallback for non-ASCII or
     heavily-modified names; runs an in-memory TF-IDF over 3-grams.

Every alias decision is persisted to `tool_aliases` with the method
that produced it and a confidence score, so a human can audit and
override individual mappings.
"""

from latentspec.canonicalization.canonicalizer import (
    AliasDecision,
    CanonicalisationResult,
    ToolCanonicalizer,
    canonical_form,
)

__all__ = [
    "AliasDecision",
    "CanonicalisationResult",
    "ToolCanonicalizer",
    "canonical_form",
]
