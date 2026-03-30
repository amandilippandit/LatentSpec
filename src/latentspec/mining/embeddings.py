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
        for feats in doc_features:
            df.update(set(feats))
        vocab = {term: idx for idx, term in enumerate(sorted(df))}
        if not vocab:
            return np.zeros((len(texts), 1))

        n_docs = len(texts)
        idf = np.zeros(len(vocab))
        for term, freq in df.items():
            idf[vocab[term]] = math.log((1 + n_docs) / (1 + freq)) + 1.0

        matrix = np.zeros((n_docs, len(vocab)), dtype=np.float64)
        for d, feats in enumerate(doc_features):
            tf = Counter(feats)
            length = max(1, sum(tf.values()))
            for term, c in tf.items():
                weight = (c / length) * idf[vocab[term]]
                if term.startswith("__ch__"):
                    weight *= self._char_trigram_weight
                matrix[d, vocab[term]] = weight

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


@dataclass
class _UnionFind:
    parent: list[int]

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Stable: smaller index becomes root
        self.parent[max(ra, rb)] = min(ra, rb)


# ---------------- Public clustering API ----------------------------------


def cluster_by_similarity(
    descriptions: Sequence[str],
    *,
    threshold: float = 0.7,
    backend: EmbeddingBackend | None = None,
) -> list[list[int]]:
    """Group descriptions by cosine similarity above `threshold`.

    Returns a list of clusters, each cluster being a list of original
    indices.
    """
    if not descriptions:
        return []
    backend = backend or TfidfBackend()
    matrix = backend.embed(list(descriptions))
    if matrix.shape[1] == 0 or matrix.size == 0:
        return [[i] for i in range(len(descriptions))]

    sims = matrix @ matrix.T

    uf = _UnionFind(len(descriptions))
    n = len(descriptions)
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                uf.union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    return list(groups.values())


def cluster_candidates_by_type_and_similarity(
    candidates: Iterable[InvariantCandidate],
    *,
    threshold: float = 0.7,
    backend: EmbeddingBackend | None = None,
) -> list[list[int]]:
    """Two candidates can only co-cluster when their `type` matches.

    This avoids clustering an ordering rule and a statistical rule even if
    their descriptions happen to share TF-IDF surface vocabulary.
    """
    cands = list(candidates)
    by_type: dict[str, list[int]] = defaultdict(list)
    for idx, c in enumerate(cands):
        by_type[c.type.value].append(idx)

    clusters: list[list[int]] = []
    for type_value, indices in by_type.items():
        descriptions = [cands[i].description for i in indices]
        sub_clusters = cluster_by_similarity(
            descriptions, threshold=threshold, backend=backend
        )
        for cluster in sub_clusters:
            # remap sub-cluster indices back to global
            clusters.append([indices[i] for i in cluster])
    return clusters
