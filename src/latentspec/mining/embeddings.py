"""TF-IDF + cosine similarity for semantic cross-track merge.

The cross-validation step in `mining/confidence.py` previously used a
normalized-string key to decide when two candidates were "the same" rule.
Two LLM rewordings ("agent always calls auth before db_write" vs
"the agent invokes auth prior to db_write") would not collide and would
both be persisted as separate invariants.

This module computes a sparse TF-IDF vector per candidate description and
clusters by cosine similarity using union-find. A pluggable
`EmbeddingBackend` interface lets users swap in a real sentence embedder
(e.g. BGE, Voyage, OpenAI text-embedding-3) without touching the merger.
"""

from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from latentspec.schemas.invariant import InvariantCandidate

log = logging.getLogger(__name__)


# ---------------- Tokenisation -------------------------------------------


_TOKEN_RE = re.compile(r"[a-z][a-z0-9_]+")
# Stop-words for invariant descriptions specifically — agent-vocabulary
# heavy, not a generic English list. Kept short on purpose so we don't
# accidentally drop semantically useful tokens like "tool", "call".
_INV_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "always", "never", "before", "after",
        "agent", "trace", "must", "this", "that", "with", "and", "or", "of",
        "to", "for", "from", "has", "have", "be", "been", "if", "when",
        "then", "as", "at", "by", "on", "in", "it", "its", "into",
    }
)


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _INV_STOPWORDS]


# ---------------- Backend interface --------------------------------------


class EmbeddingBackend(ABC):
    """Embed a list of strings into row-vectors. Vectors must be L2-normalized."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class TfidfBackend(EmbeddingBackend):
    """Sparse TF-IDF + character-trigram fallback for short descriptions.

    Pure unigrams under-cluster English paraphrases ("calls" vs "invokes"),
    so we additionally hash a small set of character trigrams stemmed from
    each token. This gives "auth" and "auth_token" non-zero overlap without
    adding a real stemmer dep.

    Bigrams are gated on document length — they help on longer rule
    descriptions but inject noise on short ones, so we skip them when the
    average document has fewer than `bigram_min_avg_tokens` tokens.
    """

    def __init__(
        self,
        *,
        ngram_max: int = 2,
        bigram_min_avg_tokens: int = 8,
        char_trigram_weight: float = 0.4,
    ) -> None:
        self._ngram_max = ngram_max
        self._bigram_min_avg_tokens = bigram_min_avg_tokens
        self._char_trigram_weight = char_trigram_weight

    def _word_features(self, text: str, *, use_bigrams: bool) -> list[str]:
        toks = _tokens(text)
        out: list[str] = list(toks)
        if use_bigrams:
            for i in range(len(toks) - 1):
                out.append("__bi__" + "_".join(toks[i : i + 2]))
        # Character trigrams over each token — gives "auth" and "auth_token"
        # partial overlap and absorbs minor English morphology ("call"/"calls").
        for tok in toks:
            for i in range(len(tok) - 2):
                out.append("__ch__" + tok[i : i + 3])
        return out

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0))

        avg_tokens = sum(len(_tokens(t)) for t in texts) / max(1, len(texts))
        use_bigrams = self._ngram_max >= 2 and avg_tokens >= self._bigram_min_avg_tokens

        doc_features = [self._word_features(t, use_bigrams=use_bigrams) for t in texts]
        df: Counter[str] = Counter()
