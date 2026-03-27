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
