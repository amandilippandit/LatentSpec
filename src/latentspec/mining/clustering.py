"""Workflow clustering — turn open-ended agents into N narrow agents.

Pipeline:

  1. **Vectorize** each trace into a fixed-length feature vector via
     `TraceShapeVectorizer`. The vector concatenates three views:
       a. Behavioral envelope (8 features, same as the anomaly miner).
       b. Tool-bag TF-IDF over the trace's tool calls.
       c. Keyword-bag TF-IDF over the trace's user_input tokens.
     The three views together capture *both* what the trace does and
     what triggered it.

  2. **Cluster** with k-means. We pick k automatically using silhouette
     score over a small grid (default k ∈ [2, 12]); the silhouette is
     the standard rigorous internal validity index for k-means and
     doesn't require a held-out set.

  3. **Route** new traces to the nearest cluster centroid. Mining runs
     once per cluster; the per-cluster invariant sets compose into the
     full active set, with each invariant carrying its `cluster_id`.

This is the foundation for handling open-ended agents: instead of one
brittle global mining run, we get N focused per-cluster runs where each
cluster is internally homogeneous enough for PrefixSpan / MI to work.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from latentspec.schemas.trace import (
    AgentResponseStep,
    NormalizedTrace,
    ToolCallStep,
    UserInputStep,
)

log = logging.getLogger(__name__)


# ---- vectorizer ----------------------------------------------------------


def _behavioral_features(trace: NormalizedTrace) -> list[float]:
    tool_calls = [s for s in trace.steps if isinstance(s, ToolCallStep)]
    distinct = {s.tool for s in tool_calls}
    user_input_chars = sum(
        len(s.content) for s in trace.steps if isinstance(s, UserInputStep)
    )
    response_chars = sum(
        len(s.content) for s in trace.steps if isinstance(s, AgentResponseStep)
    )
    latencies = [s.latency_ms or 0 for s in tool_calls]
    error_count = sum(
        1 for s in tool_calls if (s.result_status or "success") != "success"
    )
    return [
        float(len(trace.steps)),
        float(len(tool_calls)),
        float(len(distinct)),
        float(user_input_chars),
        float(response_chars),
        float(sum(latencies)),
        float(max(latencies) if latencies else 0),
        float(error_count),
    ]


def _tool_counts(trace: NormalizedTrace) -> Counter[str]:
    out: Counter[str] = Counter()
    for s in trace.steps:
        if isinstance(s, ToolCallStep):
            out[s.tool] += 1
    return out


def _keyword_counts(
    trace: NormalizedTrace, *, max_chars: int = 200
) -> Counter[str]:
    out: Counter[str] = Counter()
    for s in trace.steps:
        if isinstance(s, UserInputStep):
            text = s.content[:max_chars].lower()
            for tok in text.split():
                tok = "".join(c for c in tok if c.isalnum() or c == "_")
                if len(tok) >= 3:
                    out[tok] += 1
    return out


@dataclass
class TraceShapeVectorizer:
    """Multi-view trace vectorizer with frozen vocabularies after fit().

    Call `fit()` once on the training corpus; the vectorizer freezes its
    tool / keyword vocabularies + IDF weights + behavioral-feature scaler.
    `transform()` then maps any trace to the same fixed-length space —
    crucial for routing new traces to existing centroids.
    """

    max_tool_vocab: int = 64
    max_keyword_vocab: int = 256
    behavioral_dim: int = 8

    _tool_idx: dict[str, int] = field(default_factory=dict, init=False)
    _kw_idx: dict[str, int] = field(default_factory=dict, init=False)
    _tool_idf: np.ndarray = field(default=None, init=False)  # type: ignore
    _kw_idf: np.ndarray = field(default=None, init=False)  # type: ignore
    _behavioral_scaler: StandardScaler = field(default=None, init=False)  # type: ignore

    def fit(self, traces: Sequence[NormalizedTrace]) -> "TraceShapeVectorizer":
        n = max(1, len(traces))

        # Tool vocab — top-K most common tools across the corpus.
        tool_df: Counter[str] = Counter()
        for t in traces:
            for tool in set(_tool_counts(t)):
                tool_df[tool] += 1
        top_tools = [tool for tool, _ in tool_df.most_common(self.max_tool_vocab)]
        self._tool_idx = {tool: i for i, tool in enumerate(top_tools)}

        kw_df: Counter[str] = Counter()
        for t in traces:
            for kw in set(_keyword_counts(t)):
                kw_df[kw] += 1
        top_kws = [kw for kw, _ in kw_df.most_common(self.max_keyword_vocab)]
        self._kw_idx = {kw: i for i, kw in enumerate(top_kws)}

        self._tool_idf = np.array(
            [
                math.log((1 + n) / (1 + tool_df[tool])) + 1.0
                for tool in top_tools
            ],
            dtype=np.float64,
        )
        self._kw_idf = np.array(
            [math.log((1 + n) / (1 + kw_df[kw])) + 1.0 for kw in top_kws],
            dtype=np.float64,
        )

        # Behavioral feature scaler
        bf = np.asarray([_behavioral_features(t) for t in traces], dtype=np.float64)
        if bf.size == 0:
            bf = np.zeros((1, self.behavioral_dim))
        self._behavioral_scaler = StandardScaler()
        self._behavioral_scaler.fit(bf)
        return self

    def transform(self, traces: Sequence[NormalizedTrace]) -> np.ndarray:
        if self._behavioral_scaler is None:
            raise RuntimeError("call .fit() first")

        n_tools = len(self._tool_idx)
        n_kws = len(self._kw_idx)
        dim = self.behavioral_dim + n_tools + n_kws
        out = np.zeros((len(traces), dim), dtype=np.float64)

        bf = np.asarray(
            [_behavioral_features(t) for t in traces], dtype=np.float64
        )
        if bf.size > 0:
            bf_scaled = self._behavioral_scaler.transform(bf)
        else:
            bf_scaled = np.zeros((0, self.behavioral_dim))

        for i, trace in enumerate(traces):
            out[i, : self.behavioral_dim] = bf_scaled[i]

            # Tool TF-IDF
            tool_counts = _tool_counts(trace)
            tot = sum(tool_counts.values()) or 1
            for tool, c in tool_counts.items():
                idx = self._tool_idx.get(tool)
                if idx is None:
                    continue
                out[i, self.behavioral_dim + idx] = (c / tot) * self._tool_idf[idx]

            # Keyword TF-IDF
            kw_counts = _keyword_counts(trace)
            tot_kw = sum(kw_counts.values()) or 1
            for kw, c in kw_counts.items():
                idx = self._kw_idx.get(kw)
                if idx is None:
                    continue
                out[i, self.behavioral_dim + n_tools + idx] = (
                    (c / tot_kw) * self._kw_idf[idx]
                )

        # L2 normalize so cosine and Euclidean agree as cluster distances
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def fit_transform(self, traces: Sequence[NormalizedTrace]) -> np.ndarray:
        return self.fit(traces).transform(traces)


# ---- clustering ----------------------------------------------------------


@dataclass
class WorkflowClustering:
    vectorizer: TraceShapeVectorizer
    kmeans: KMeans
    labels: np.ndarray
    silhouette: float
    k: int
    cluster_sizes: dict[int, int]
    centroids: np.ndarray

    def predict(self, traces: Sequence[NormalizedTrace]) -> np.ndarray:
        X = self.vectorizer.transform(traces)
        return self.kmeans.predict(X)


def _silhouette_safe(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette is undefined for k=1 or when any cluster has ≤ 1 point."""
    unique = set(labels.tolist())
    if len(unique) < 2:
        return -1.0
    sizes = Counter(labels.tolist())
    if any(c < 2 for c in sizes.values()):
        return -1.0
    try:
        return float(silhouette_score(X, labels, metric="cosine"))
    except Exception:
        return -1.0


def cluster_workflows(
    traces: Sequence[NormalizedTrace],
    *,
    k_min: int = 2,
    k_max: int = 12,
    min_traces_per_cluster: int = 8,
    random_state: int = 17,
    vectorizer: TraceShapeVectorizer | None = None,
) -> WorkflowClustering:
    """Cluster traces into workflow families. Returns the fitted artifact.

    Picks k via silhouette score over `[k_min, k_max]`. Falls back to a
    single-cluster result when fewer than `min_traces_per_cluster * k_min`
    traces are available — too few to clustering cleanly.
    """
    if not traces:
        raise ValueError("cluster_workflows: empty trace set")

    vec = vectorizer or TraceShapeVectorizer()
    X = vec.fit_transform(traces)
    n = X.shape[0]

    # Cap k_max by what the corpus can support
    k_cap = max(k_min, min(k_max, n // max(1, min_traces_per_cluster)))
    if k_cap < k_min or n < k_min * min_traces_per_cluster:
        # One-cluster fallback
        labels = np.zeros(n, dtype=int)
        km = KMeans(n_clusters=1, random_state=random_state, n_init=4)
        km.fit(X)
        return WorkflowClustering(
            vectorizer=vec,
            kmeans=km,
            labels=labels,
            silhouette=0.0,
            k=1,
            cluster_sizes={0: n},
            centroids=km.cluster_centers_,
        )

    best: WorkflowClustering | None = None
    for k in range(k_min, k_cap + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=4)
        labels = km.fit_predict(X)
        score = _silhouette_safe(X, labels)
        if best is None or score > best.silhouette:
            best = WorkflowClustering(
                vectorizer=vec,
                kmeans=km,
                labels=labels,
                silhouette=score,
                k=k,
                cluster_sizes=dict(Counter(labels.tolist())),
                centroids=km.cluster_centers_,
            )
    assert best is not None
    return best


def split_by_cluster(
    traces: Sequence[NormalizedTrace], labels: np.ndarray
) -> dict[int, list[NormalizedTrace]]:
    out: dict[int, list[NormalizedTrace]] = {}
    for trace, label in zip(traces, labels.tolist(), strict=True):
        out.setdefault(int(label), []).append(trace)
    return out
